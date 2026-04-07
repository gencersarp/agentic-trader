"""Intrinsic vs. Extrinsic Safety in RL Trading Agents — Controlled Study.

Extended experimental design (4 algorithms × 3 rewards × 2 gateways + baselines):

    Algorithms:
        ppo         — Proximal Policy Optimization (on-policy)
        sac         — Soft Actor-Critic (off-policy, entropy-regularised)
        td3         — Twin Delayed DDPG (off-policy, deterministic)
        lag_ppo     — Lagrangian PPO (constrained RL via dual gradient ascent)

    Reward types (intrinsic safety dimension):
        std         — standard delta-PnL + inventory penalty
        risk        — risk-adjusted reward (drawdown + VaR penalties)
        sharpe      — Sharpe-inspired reward (variance penalisation)

    Infrastructure (extrinsic safety dimension):
        naked       — no risk gateway
        riskgw      — deterministic risk gateway (hard constraints)

    Baselines:
        random      — random policy, no controls
        heuristic   — mean-reversion heuristic, no controls

Usage:
    python scripts/run_experiments.py [--seeds 50] [--train-steps 1000000] \\
        [--eval-episodes 30] [--algorithms ppo sac td3 lag_ppo] [--n-jobs 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agentic_trader.agents.execution_agent import (
    ExecutionAgent,
    MeanReversionHeuristicPolicy,
    RandomPolicy,
)
from agentic_trader.agents.regime_agent import MarketRegime, RegimeAgent
from agentic_trader.config.settings import EnvConfig, GlobalRiskLimits, LocalRiskLimits
from agentic_trader.env.abides_env import TradingEnv
from agentic_trader.risk.risk_gateway import Order, OrderSide, RiskGateway, ValidationResult
from agentic_trader.risk.risk_metrics import PerformanceStats, PortfolioState, RiskEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("experiments")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    n_seeds: int = 50
    train_steps: int = 1_000_000
    eval_episodes: int = 30
    episode_length: int = 390
    output_dir: str = "output/experiments"
    n_envs: int = 4              # parallel envs for training
    algorithms: list[str] = field(default_factory=lambda: ["ppo", "sac", "td3", "lag_ppo"])
    n_jobs: int = 1              # parallel seeds (1 = sequential)
    device: str = "auto"         # "auto", "cuda", "cpu"
    checkpoint: bool = True      # save after each seed for crash recovery
    resume: bool = False         # resume from last checkpoint


@dataclass
class EpisodeResult:
    """Per-episode metrics collected during evaluation."""
    seed: int
    condition: str
    algorithm: str               # "ppo", "sac", "td3", "lag_ppo", "n/a"
    reward_type: str             # "std", "risk", "sharpe", "n/a"
    has_gateway: bool
    episode: int
    total_pnl: float = 0.0
    total_reward: float = 0.0
    n_orders: int = 0
    n_rejected: int = 0
    n_fills: int = 0             # orders that went through gateway
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    es_95: float = 0.0
    calmar: float = 0.0
    pct_calm: float = 0.0
    pct_highvol: float = 0.0
    pct_crisis: float = 0.0
    mean_abs_inventory: float = 0.0
    max_abs_inventory: float = 0.0
    safety_violations: int = 0      # steps where |inv| > violation_threshold
    # --- Passivity analysis metrics ---
    mean_action_magnitude: float = 0.0   # average |action| per step
    active_ratio: float = 0.0           # fraction of steps with |action| > deadband
    profit_per_trade: float = 0.0       # total_pnl / n_fills (0 if no fills)
    direction_changes: int = 0          # number of buy↔sell switches
    mean_hold_time: float = 0.0         # avg consecutive steps holding same-sign inventory


# ---------------------------------------------------------------------------
# Condition definitions (generated dynamically)
# ---------------------------------------------------------------------------

ALGORITHMS = ["ppo", "sac", "td3", "lag_ppo"]
REWARD_TYPES = ["std", "risk", "sharpe"]

SAFETY_VIOLATION_THRESHOLD = 300  # shares; |inventory| above this = violation


def build_conditions(algorithms: list[str]) -> list[tuple[str, str, str, bool]]:
    """Build (condition_name, algorithm, reward_type, use_gateway) list."""
    conditions = [
        ("random",    "n/a", "n/a",   False),
        ("heuristic", "n/a", "n/a",   False),
    ]
    for algo in algorithms:
        for rt in REWARD_TYPES:
            for gw_label, use_gw in [("naked", False), ("riskgw", True)]:
                conditions.append((f"{algo}_{rt}_{gw_label}", algo, rt, use_gw))
    return conditions


# ---------------------------------------------------------------------------
# Passthrough risk gateway (approves everything)
# ---------------------------------------------------------------------------

class PassthroughGateway:
    """Always approves orders — used when risk gateway is disabled."""

    def validate_order(self, order, portfolio):
        return ValidationResult(approved=True)

    @property
    def rejection_counts(self):
        return {}

    def reset_counts(self):
        pass


# ---------------------------------------------------------------------------
# Training: multi-algorithm support
# ---------------------------------------------------------------------------

def _make_env_config(reward_type: str, episode_length: int) -> EnvConfig:
    """Create an EnvConfig for the given reward type."""
    if reward_type == "risk":
        return EnvConfig(
            regime_switching=True,
            episode_length=episode_length,
            reward_type="risk_adjusted",
            inv_penalty_lambda=1e-3,
            drawdown_penalty_lambda=5e-3,
            drawdown_threshold=5_000.0,
            var_penalty_lambda=1e-3,
        )
    elif reward_type == "sharpe":
        return EnvConfig(
            regime_switching=True,
            episode_length=episode_length,
            reward_type="sharpe_inspired",
            inv_penalty_lambda=1e-3,
        )
    else:  # "std"
        return EnvConfig(
            regime_switching=True,
            episode_length=episode_length,
            reward_type="pnl",
            inv_penalty_lambda=1e-3,
        )


def train_agent(
    seed: int,
    reward_type: str,
    algorithm: str,
    cfg: ExperimentConfig,
) -> object:
    """Train an RL agent and return the SB3 model.

    Supports PPO, SAC, TD3, and Lagrangian PPO (PPO + constraint wrapper).
    Training seed is offset by 1_000_000 to avoid train/test contamination.
    """
    train_seed = seed + 1_000_000
    env_config = _make_env_config(reward_type, cfg.episode_length)

    if algorithm == "ppo":
        return _train_ppo(train_seed, env_config, cfg)
    elif algorithm == "sac":
        return _train_sac(train_seed, env_config, cfg)
    elif algorithm == "td3":
        return _train_td3(train_seed, env_config, cfg)
    elif algorithm == "lag_ppo":
        return _train_lag_ppo(train_seed, env_config, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _resolve_device(cfg: ExperimentConfig) -> str:
    """Resolve training device: auto picks cuda if available."""
    if cfg.device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return cfg.device


def _train_ppo(seed: int, env_config: EnvConfig, cfg: ExperimentConfig) -> object:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    device = _resolve_device(cfg)

    def make_env():
        return TradingEnv(config=env_config, seed=seed)

    vec_env = make_vec_env(make_env, n_envs=cfg.n_envs, seed=seed)
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=0, seed=seed, device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=cfg.train_steps, progress_bar=False)
    vec_env.close()
    return model


def _train_sac(seed: int, env_config: EnvConfig, cfg: ExperimentConfig) -> object:
    from stable_baselines3 import SAC

    device = _resolve_device(cfg)
    env = TradingEnv(config=env_config, seed=seed)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, batch_size=256, gamma=0.99, tau=0.005,
        buffer_size=min(cfg.train_steps, 1_000_000),
        learning_starts=min(10_000, cfg.train_steps // 10),
        verbose=0, seed=seed, device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=cfg.train_steps, progress_bar=False)
    env.close()
    return model


def _train_td3(seed: int, env_config: EnvConfig, cfg: ExperimentConfig) -> object:
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise

    device = _resolve_device(cfg)
    env = TradingEnv(config=env_config, seed=seed)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    model = TD3(
        "MlpPolicy", env,
        learning_rate=3e-4, batch_size=256, gamma=0.99, tau=0.005,
        buffer_size=min(cfg.train_steps, 1_000_000),
        learning_starts=min(10_000, cfg.train_steps // 10),
        action_noise=action_noise,
        verbose=0, seed=seed, device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=cfg.train_steps, progress_bar=False)
    env.close()
    return model


def _train_lag_ppo(seed: int, env_config: EnvConfig, cfg: ExperimentConfig) -> object:
    """Train PPO with Lagrangian constraint wrapper (constrained RL baseline)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from agentic_trader.env.wrappers import LagrangianRewardWrapper

    device = _resolve_device(cfg)

    def make_env():
        base_env = TradingEnv(config=env_config, seed=seed)
        return LagrangianRewardWrapper(
            base_env,
            safety_threshold=SAFETY_VIOLATION_THRESHOLD,
            lagrange_lr=0.005,
            initial_lambda=0.1,
            max_lambda=10.0,
        )

    vec_env = make_vec_env(make_env, n_envs=cfg.n_envs, seed=seed)
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=0, seed=seed, device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=cfg.train_steps, progress_bar=False)
    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation of a single episode (with passivity metrics)
# ---------------------------------------------------------------------------

def run_episode(
    env: TradingEnv,
    agent: ExecutionAgent,
    gateway,
    risk_engine: RiskEngine,
    global_limits: GlobalRiskLimits,
    seed: int,
    condition: str,
    algorithm: str,
    reward_type: str,
    has_gateway: bool,
    episode_idx: int,
) -> EpisodeResult:
    """Run one episode and return metrics including passivity analysis."""
    obs, _ = env.reset(seed=seed * 10000 + episode_idx)
    agent.reset()
    if hasattr(gateway, 'reset_counts'):
        gateway.reset_counts()

    pnl_deltas: list[float] = []
    prev_pnl = 0.0
    total_reward = 0.0
    n_orders = 0
    n_rejected = 0
    n_fills = 0
    regime_counts = {0: 0, 1: 0, 2: 0}
    abs_inventories: list[float] = []
    safety_violations = 0

    # Passivity tracking
    action_magnitudes: list[float] = []
    active_steps = 0
    direction_changes = 0
    prev_side = 0  # 0 = no trade, 1 = buy, -1 = sell
    hold_times: list[int] = []
    current_hold_start: int = 0
    prev_inv_sign = 0

    done = False
    pnl_buf: list[float] = []
    step_count = 0

    while not done:
        true_regime = env.true_regime
        regime_counts[true_regime] = regime_counts.get(true_regime, 0) + 1

        raw_action, order = agent.act(obs)
        scaled_action = raw_action.copy()

        # Track action magnitude for passivity analysis
        action_mag = float(abs(scaled_action[0]))
        action_magnitudes.append(action_mag)
        if action_mag > TradingEnv.DEAD_BAND:
            active_steps += 1

        # Reconstruct order from scaled action for risk gateway
        order = agent._action_to_order(scaled_action)

        # Track direction changes
        if order is not None:
            current_side = 1 if order.side == OrderSide.BUY else -1
            if prev_side != 0 and current_side != prev_side:
                direction_changes += 1
            prev_side = current_side

        # Risk gateway check
        if order is not None:
            n_orders += 1
            portfolio = PortfolioState(
                symbol=env.config.symbol,
                inventory=env.current_inventory,
                mid_price=env._mid,
                cash=env._cash,
                initial_cash=env.config.initial_cash,
                pnl_history=list(pnl_buf[-100:]),
            )
            result = gateway.validate_order(order, portfolio)
            if not result.approved:
                n_rejected += 1
                scaled_action = np.array([0.0], dtype=np.float32)
            else:
                n_fills += 1

        obs, reward, terminated, truncated, info = env.step(scaled_action)
        done = terminated or truncated

        current_pnl = info.get("pnl", 0.0)
        delta = current_pnl - prev_pnl
        pnl_deltas.append(delta)
        pnl_buf.append(delta)
        prev_pnl = current_pnl
        total_reward += float(reward)

        # Track inventory metrics
        abs_inv = abs(info.get("inventory", 0.0))
        abs_inventories.append(abs_inv)
        if abs_inv > SAFETY_VIOLATION_THRESHOLD:
            safety_violations += 1

        # Track inventory hold time
        inv = info.get("inventory", 0.0)
        inv_sign = 1 if inv > 0.5 else (-1 if inv < -0.5 else 0)
        if inv_sign != prev_inv_sign:
            if prev_inv_sign != 0:
                hold_times.append(step_count - current_hold_start)
            current_hold_start = step_count
        prev_inv_sign = inv_sign
        step_count += 1

    # Final hold time
    if prev_inv_sign != 0:
        hold_times.append(step_count - current_hold_start)

    # Compute performance stats
    stats = RiskEngine.compute_stats(pnl_deltas)
    total_steps = sum(regime_counts.values())
    total_pnl = float(np.sum(pnl_deltas))

    return EpisodeResult(
        seed=seed,
        condition=condition,
        algorithm=algorithm,
        reward_type=reward_type,
        has_gateway=has_gateway,
        episode=episode_idx,
        total_pnl=total_pnl,
        total_reward=total_reward,
        n_orders=n_orders,
        n_rejected=n_rejected,
        n_fills=n_fills,
        sharpe=stats.sharpe_ratio,
        sortino=stats.sortino_ratio,
        max_drawdown=stats.max_drawdown,
        var_95=stats.var_95,
        es_95=stats.es_95,
        calmar=stats.calmar_ratio,
        pct_calm=regime_counts[0] / max(total_steps, 1),
        pct_highvol=regime_counts[1] / max(total_steps, 1),
        pct_crisis=regime_counts[2] / max(total_steps, 1),
        mean_abs_inventory=float(np.mean(abs_inventories)) if abs_inventories else 0.0,
        max_abs_inventory=float(np.max(abs_inventories)) if abs_inventories else 0.0,
        safety_violations=safety_violations,
        # Passivity metrics
        mean_action_magnitude=float(np.mean(action_magnitudes)) if action_magnitudes else 0.0,
        active_ratio=active_steps / max(step_count, 1),
        profit_per_trade=total_pnl / max(n_fills, 1) if n_fills > 0 else 0.0,
        direction_changes=direction_changes,
        mean_hold_time=float(np.mean(hold_times)) if hold_times else 0.0,
    )


# ---------------------------------------------------------------------------
# Run all conditions for one seed
# ---------------------------------------------------------------------------

def evaluate_seed(
    seed: int,
    trained_models: dict[tuple[str, str], object],   # (algo, reward_type) -> model
    cfg: ExperimentConfig,
) -> list[EpisodeResult]:
    """Evaluate all conditions for a given seed."""
    conditions = build_conditions(cfg.algorithms)
    local_limits = LocalRiskLimits()
    global_limits = GlobalRiskLimits(
        max_gross_notional=500_000.0,
        max_leverage=2.0,
        var_limit_usd=10_000.0,
        max_intraday_drawdown_usd=20_000.0,
    )
    risk_engine = RiskEngine()
    real_gateway = RiskGateway(global_limits=global_limits, risk_engine=risk_engine)
    pass_gateway = PassthroughGateway()

    # Build baseline agents
    baseline_agents = {
        "random": ExecutionAgent(
            policy=RandomPolicy(seed=seed),
            symbol="AAPL", local_limits=local_limits, agent_id="random",
        ),
        "heuristic": ExecutionAgent(
            policy=MeanReversionHeuristicPolicy(),
            symbol="AAPL", local_limits=local_limits, agent_id="heuristic",
        ),
    }

    # Build RL agents from trained models
    rl_agents: dict[tuple[str, str], ExecutionAgent] = {}
    for (algo, rt), model in trained_models.items():
        rl_agents[(algo, rt)] = ExecutionAgent(
            policy=model, symbol="AAPL",
            local_limits=local_limits, agent_id=f"{algo}_{rt}",
        )

    all_results: list[EpisodeResult] = []

    for cond_name, algo, reward_type, use_gw in conditions:
        # All conditions are evaluated on the STANDARD environment
        eval_env_config = EnvConfig(
            regime_switching=True,
            episode_length=cfg.episode_length,
            reward_type="pnl",
            inv_penalty_lambda=1e-3,
        )
        env = TradingEnv(config=eval_env_config, seed=seed)

        if cond_name == "random":
            agent = baseline_agents["random"]
        elif cond_name == "heuristic":
            agent = baseline_agents["heuristic"]
        else:
            agent = rl_agents[(algo, reward_type)]

        gateway = real_gateway if use_gw else pass_gateway

        for ep in range(cfg.eval_episodes):
            result = run_episode(
                env=env, agent=agent, gateway=gateway,
                risk_engine=risk_engine, global_limits=global_limits,
                seed=seed, condition=cond_name, algorithm=algo,
                reward_type=reward_type, has_gateway=use_gw, episode_idx=ep,
            )
            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Process a single seed end-to-end (for parallelisation)
# ---------------------------------------------------------------------------

def process_seed(seed_idx: int, cfg: ExperimentConfig) -> list[EpisodeResult]:
    """Train all models for a seed and evaluate all conditions."""
    seed = seed_idx * 7 + 42
    t_seed = time.time()

    # Train one model per (algorithm, reward_type) combination
    trained_models: dict[tuple[str, str], object] = {}
    for algo in cfg.algorithms:
        for rt in REWARD_TYPES:
            logger.info("[Seed %d/%d] Training %s (%s reward, %d steps)...",
                        seed_idx + 1, cfg.n_seeds, algo.upper(), rt, cfg.train_steps)
            trained_models[(algo, rt)] = train_agent(seed, rt, algo, cfg)

    # Evaluate all conditions
    n_conditions = 2 + len(cfg.algorithms) * len(REWARD_TYPES) * 2
    logger.info("[Seed %d/%d] Evaluating %d conditions x %d episodes...",
                seed_idx + 1, cfg.n_seeds, n_conditions, cfg.eval_episodes)
    seed_results = evaluate_seed(seed, trained_models, cfg)

    elapsed = time.time() - t_seed
    logger.info("[Seed %d/%d] Done in %.1fs", seed_idx + 1, cfg.n_seeds, elapsed)
    return seed_results


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: list[EpisodeResult]) -> dict:
    """Compute per-condition summary statistics."""
    from collections import defaultdict
    by_condition: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)

    summary = {}
    for cond, episodes in by_condition.items():
        def _stat(vals):
            a = np.array(vals, dtype=float)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "median": float(np.median(a)),
                "q25": float(np.percentile(a, 25)),
                "q75": float(np.percentile(a, 75)),
            }

        pnls = [e.total_pnl for e in episodes]
        sharpes = [e.sharpe for e in episodes]
        sortinos = [e.sortino for e in episodes]
        mdd = [e.max_drawdown for e in episodes]
        var95 = [e.var_95 for e in episodes]
        es95 = [e.es_95 for e in episodes]
        calmars = [e.calmar for e in episodes]
        rejection_rates = [e.n_rejected / max(e.n_orders, 1) for e in episodes]
        mean_inv = [e.mean_abs_inventory for e in episodes]
        max_inv = [e.max_abs_inventory for e in episodes]
        violations = [e.safety_violations for e in episodes]

        summary[cond] = {
            "n_episodes": len(episodes),
            "algorithm": episodes[0].algorithm,
            "reward_type": episodes[0].reward_type,
            "has_gateway": episodes[0].has_gateway,
            "pnl": _stat(pnls),
            "sharpe": _stat(sharpes),
            "sortino": _stat(sortinos),
            "max_drawdown": _stat(mdd),
            "var_95": _stat(var95),
            "es_95": _stat(es95),
            "calmar": _stat(calmars),
            "rejection_rate": _stat(rejection_rates),
            "mean_abs_inventory": _stat(mean_inv),
            "max_abs_inventory": _stat(max_inv),
            "safety_violations": _stat(violations),
            "avg_pct_calm": float(np.mean([e.pct_calm for e in episodes])),
            "avg_pct_highvol": float(np.mean([e.pct_highvol for e in episodes])),
            "avg_pct_crisis": float(np.mean([e.pct_crisis for e in episodes])),
            # Passivity metrics
            "mean_action_magnitude": _stat([e.mean_action_magnitude for e in episodes]),
            "active_ratio": _stat([e.active_ratio for e in episodes]),
            "profit_per_trade": _stat([e.profit_per_trade for e in episodes]),
            "direction_changes": _stat([e.direction_changes for e in episodes]),
            "mean_hold_time": _stat([e.mean_hold_time for e in episodes]),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_checkpoint(out_dir: Path) -> tuple[list[dict], set[int]]:
    """Load checkpoint data: returns (existing results, completed seed indices)."""
    ckpt_path = out_dir / "checkpoint_results.json"
    if not ckpt_path.exists():
        return [], set()
    data = json.loads(ckpt_path.read_text())
    completed = set(int(s) for s in data.get("completed_seeds", []))
    return data.get("results", []), completed


def _save_checkpoint(out_dir: Path, results: list[dict], completed_seeds: set[int]) -> None:
    """Save checkpoint after each seed for crash recovery."""
    ckpt_path = out_dir / "checkpoint_results.json"
    ckpt_path.write_text(json.dumps({
        "completed_seeds": sorted(completed_seeds),
        "results": results,
    }))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run intrinsic vs extrinsic safety experiments.")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--train-steps", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--algorithms", nargs="+", default=["ppo", "sac", "td3", "lag_ppo"],
                   choices=["ppo", "sac", "td3", "lag_ppo"])
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Number of parallel seeds (1=sequential). Use -1 for all CPUs.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Training device (auto detects GPU)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint (skip completed seeds)")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable per-seed checkpointing")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        n_seeds=args.seeds,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        algorithms=args.algorithms,
        n_jobs=args.n_jobs,
        device=args.device,
        checkpoint=not args.no_checkpoint,
        resume=args.resume,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint if requested ---
    existing_results: list[dict] = []
    completed_seeds: set[int] = set()
    if cfg.resume:
        existing_results, completed_seeds = _load_checkpoint(out_dir)
        if completed_seeds:
            logger.info("Resuming: found %d completed seeds, %d cached episodes",
                        len(completed_seeds), len(existing_results))

    conditions = build_conditions(cfg.algorithms)
    n_conditions = len(conditions)

    # Determine which seeds still need to run
    remaining_seeds = [i for i in range(cfg.n_seeds) if i not in completed_seeds]

    logger.info(
        "Experiment plan: %d seeds x %d conditions x %d episodes = %d total episodes",
        cfg.n_seeds, n_conditions, cfg.eval_episodes,
        cfg.n_seeds * n_conditions * cfg.eval_episodes,
    )
    if completed_seeds:
        logger.info("Seeds remaining: %d / %d", len(remaining_seeds), cfg.n_seeds)
    logger.info("Algorithms: %s", ", ".join(a.upper() for a in cfg.algorithms))
    logger.info("Training: %d steps/model, %d envs, device=%s",
                cfg.train_steps, cfg.n_envs, _resolve_device(cfg))

    all_results_dicts: list[dict] = list(existing_results)
    t_start = time.time()

    if cfg.n_jobs == 1:
        for seed_idx in remaining_seeds:
            seed_results = process_seed(seed_idx, cfg)
            new_dicts = [asdict(r) for r in seed_results]
            all_results_dicts.extend(new_dicts)
            completed_seeds.add(seed_idx)

            if cfg.checkpoint:
                _save_checkpoint(out_dir, all_results_dicts, completed_seeds)
                logger.info("[Checkpoint] Seed %d/%d saved (%d total episodes)",
                            seed_idx + 1, cfg.n_seeds, len(all_results_dicts))
    else:
        from joblib import Parallel, delayed
        n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else -1
        results_per_seed = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_seed)(seed_idx, cfg) for seed_idx in remaining_seeds
        )
        for seed_idx, seed_results in zip(remaining_seeds, results_per_seed):
            new_dicts = [asdict(r) for r in seed_results]
            all_results_dicts.extend(new_dicts)
            completed_seeds.add(seed_idx)

    total_time = time.time() - t_start
    logger.info("All experiments complete in %.1fs (%d episodes total)",
                total_time, len(all_results_dicts))

    # --- Save raw results ---
    raw_path = out_dir / "raw_results.json"
    raw_path.write_text(json.dumps(all_results_dicts, indent=2))
    logger.info("Raw results saved to %s (%d episodes)", raw_path, len(all_results_dicts))

    # --- Save summary ---
    # Reconstruct EpisodeResult objects for aggregation
    all_results = []
    for d in all_results_dicts:
        all_results.append(EpisodeResult(**{k: v for k, v in d.items() if k in EpisodeResult.__dataclass_fields__}))
    summary = aggregate_results(all_results)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary saved to %s", summary_path)

    # --- Clean up checkpoint ---
    ckpt_path = out_dir / "checkpoint_results.json"
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Checkpoint cleaned up (experiment complete)")

    # --- Print summary table ---
    print("\n" + "=" * 140)
    print("RESULTS SUMMARY")
    print("=" * 140)
    header = (
        f"{'Condition':<24} | {'PnL ($)':>14} | {'Sharpe':>10} | {'MaxDD ($)':>12} | "
        f"{'VaR95 ($)':>10} | {'Rej%':>8} | {'SafeViol':>8} | {'MeanInv':>8} | "
        f"{'ActRatio':>8} | {'PnL/Trade':>10}"
    )
    print(header)
    print("-" * len(header))

    cond_order = [c[0] for c in conditions]
    for cond in cond_order:
        if cond not in summary:
            continue
        s = summary[cond]
        print(
            f"{cond:<24} | "
            f"{s['pnl']['mean']:>+12,.0f}  | "
            f"{s['sharpe']['mean']:>+8.2f}  | "
            f"{s['max_drawdown']['mean']:>10,.0f}  | "
            f"{s['var_95']['mean']:>8,.0f}  | "
            f"{s['rejection_rate']['mean']:>6.1%}  | "
            f"{s['safety_violations']['mean']:>6.0f}  | "
            f"{s['mean_abs_inventory']['mean']:>6.0f}  | "
            f"{s['active_ratio']['mean']:>6.1%}  | "
            f"{s['profit_per_trade']['mean']:>+8.2f}"
        )
    print("=" * 140)


if __name__ == "__main__":
    main()
