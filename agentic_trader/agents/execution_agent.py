"""ExecutionAgent — wraps a Stable-Baselines3 policy.

Responsibilities:
  * Translate raw RL policy outputs into validated, size-clipped orders.
  * Apply *local* (per-agent) constraints before passing orders to the RiskGateway.
  * Expose a clean `act(obs)` interface that returns a numpy action array and a
    structured Order object so the Orchestrator can feed both to the env and the
    gateway without duplicated logic.

Training is handled in scripts/train_execution_agent.py; this class is only
used for inference / deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from agentic_trader.config.settings import LocalRiskLimits
from agentic_trader.risk.risk_gateway import Order, OrderSide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy protocol — allows swapping SB3 for other frameworks
# ---------------------------------------------------------------------------


class PolicyProtocol:
    """Minimal interface that any policy must satisfy."""

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ExecutionAgent
# ---------------------------------------------------------------------------


class ExecutionAgent:
    """RL-policy wrapper for a single-asset execution agent.

    Args:
        policy: A Stable-Baselines3 model (or any object with a `.predict` method).
        symbol: The instrument this agent trades.
        local_limits: Per-agent position and order-size constraints.
        agent_id: Unique identifier for logging and tracing.
    """

    def __init__(
        self,
        policy: PolicyProtocol,
        symbol: str,
        local_limits: LocalRiskLimits,
        agent_id: str = "exec_agent_0",
    ):
        self.policy = policy
        self.symbol = symbol
        self.local_limits = local_limits
        self.agent_id = agent_id

        # Internal inventory tracker (mirrors environment state for constraint checks)
        self._inventory: float = 0.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, Optional[Order]]:
        """Produce a (clipped) action array and a corresponding Order.

        Returns:
            action: numpy array in [-1, 1] — passed directly to env.step().
            order:  an Order dataclass for RiskGateway validation; None if no trade.
        """
        raw_action, _ = self.policy.predict(obs, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32).flatten()

        clipped_action = self._apply_local_constraints(raw_action)

        order = self._action_to_order(clipped_action)
        return clipped_action, order

    def action_description(self, action: np.ndarray) -> str:
        """Return a human-readable label for an action vector.

        Useful for logging and dashboard display without having to decode
        the raw float each time.

        Examples: "BUY 3 units (signal=0.61)", "HOLD (signal=0.00)", "SELL 1 unit (signal=-0.22)"
        """
        a = float(np.clip(np.asarray(action, dtype=np.float32).flatten()[0], -1.0, 1.0))
        size = round(abs(a) * self.local_limits.max_order_size)
        if size == 0 or abs(a) < 1e-4:
            return f"HOLD (signal={a:.2f})"
        direction = "BUY" if a > 0 else "SELL"
        unit_label = "unit" if size == 1 else "units"
        return f"{direction} {size} {unit_label} (signal={a:.2f})"

    def update_inventory(self, delta: float) -> None:
        """Called by the Orchestrator after a fill to keep internal state in sync."""
        self._inventory += delta

    def reset(self) -> None:
        """Reset per-episode state."""
        self._inventory = 0.0

    # ------------------------------------------------------------------
    # Local constraints
    # ------------------------------------------------------------------

    def _apply_local_constraints(self, action: np.ndarray) -> np.ndarray:
        """Project action into locally admissible range.

        Checks:
          1. Clip action scalar to [-1, 1].
          2. If buying would breach max_inventory, zero out the action.
          3. If selling would breach min_inventory, zero out the action.
        """
        a = float(np.clip(action[0], -1.0, 1.0))
        side = np.sign(a)
        order_size = round(abs(a) * self.local_limits.max_order_size)

        new_inv = self._inventory + side * order_size
        if new_inv > self.local_limits.max_inventory:
            logger.debug("%s: buy clipped — inventory would reach %.0f", self.agent_id, new_inv)
            a = 0.0
        elif new_inv < self.local_limits.min_inventory:
            logger.debug("%s: sell clipped — inventory would reach %.0f", self.agent_id, new_inv)
            a = 0.0

        return np.array([a], dtype=np.float32)

    def _action_to_order(self, action: np.ndarray) -> Optional[Order]:
        """Convert a clipped scalar action into a structured Order.

        Returns None for dead-band actions (|a| < 0.05) — no order is placed.
        """
        a = float(action[0])
        dead_band = 0.05
        if abs(a) < dead_band:
            return None

        side = OrderSide.BUY if a > 0 else OrderSide.SELL
        size = max(1, round(abs(a) * self.local_limits.max_order_size))
        return Order(symbol=self.symbol, side=side, size=size, agent_id=self.agent_id)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        symbol: str,
        local_limits: LocalRiskLimits,
        agent_id: str = "exec_agent_0",
    ) -> "ExecutionAgent":
        """Load a Stable-Baselines3 PPO/TD3 policy from disk."""
        try:
            from stable_baselines3 import PPO, TD3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required: pip install stable-baselines3") from exc

        path = Path(path)
        # Try PPO first, then TD3
        for AlgoCls in (PPO, TD3):
            try:
                model = AlgoCls.load(path)
                logger.info("Loaded %s policy from %s", AlgoCls.__name__, path)
                return cls(policy=model, symbol=symbol, local_limits=local_limits, agent_id=agent_id)
            except Exception:
                continue
        raise ValueError(f"Could not load policy from {path} as PPO or TD3.")

    def save(self, path: str | Path) -> None:
        """Save the underlying SB3 model (only works if policy is an SB3 model)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(str(path))  # type: ignore[attr-defined]
        logger.info("Policy saved to %s", path)


# ---------------------------------------------------------------------------
# Random / heuristic policy fallbacks (useful for testing without training)
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Random action policy — used for smoke-testing the pipeline."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        action = self._rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        return action, None


class MeanReversionHeuristicPolicy:
    """Simple rule-based policy: buy on negative short-term return, sell on positive.

    obs layout (from TradingEnv):
        [bid, ask, bid_depth, ask_depth, mid, spread,
         ret_t-1 .. ret_t-5,
         inv_norm, cash_pnl_norm, step_norm]
    """

    RETURN_START_IDX = 6    # index of ret_t-1 in obs
    N_RETURNS = 5
    INV_IDX = 11
    THRESHOLD = 0.003       # ~30 bps z-score threshold

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = obs.flatten()
        returns = obs[self.RETURN_START_IDX : self.RETURN_START_IDX + self.N_RETURNS]
        avg_ret = float(np.mean(returns))
        inv_norm = float(obs[self.INV_IDX])

        # Mean-reversion signal: fade recent direction, damped by inventory
        signal = -avg_ret / (self.THRESHOLD + abs(avg_ret))
        signal *= max(0.0, 1.0 - abs(inv_norm))   # reduce sizing near inventory limit

        action = np.array([float(np.clip(signal, -1.0, 1.0))], dtype=np.float32)
        return action, None
