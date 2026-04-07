"""Centralised configuration via dataclasses.

All tunable parameters live here so that scripts only import Settings and
never hard-code values.  Each sub-config can be overridden independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Environment / simulation
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Parameters for the LOB simulation environment."""

    symbol: str = "AAPL"
    episode_length: int = 390        # timesteps per episode (≈ minutes in a trading day)
    tick_size: float = 0.01
    initial_cash: float = 1_000_000.0
    max_inventory: int = 1_000       # shares; used for action scaling and obs normalisation
    annual_vol: float = 0.20         # GBM volatility for synthetic price process
    annual_drift: float = 0.0        # GBM drift
    regime_switching: bool = True    # enable 3-state regime-switching volatility

    # Reward shaping configuration
    reward_type: str = "pnl"         # "pnl", "risk_adjusted", "sharpe_inspired"
    inv_penalty_lambda: float = 1e-3  # inventory penalty coefficient
    drawdown_penalty_lambda: float = 5e-3  # penalty for exceeding drawdown threshold
    drawdown_threshold: float = 5_000.0    # USD drawdown before penalty kicks in
    var_penalty_lambda: float = 1e-3       # penalty on rolling VaR estimate


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------


@dataclass
class LocalRiskLimits:
    """Per-agent constraints applied inside ExecutionAgent._apply_local_constraints."""

    max_order_size: int = 100        # shares per timestep
    max_inventory: int = 500         # max long inventory
    min_inventory: int = -500        # max short inventory


@dataclass
class GlobalRiskLimits:
    """Firm-wide limits enforced by RiskGateway (deterministic, out-of-band)."""

    max_gross_notional: float = 500_000.0   # USD
    max_leverage: float = 2.0
    var_limit_usd: float = 10_000.0         # 1-day 95% VaR ceiling
    max_intraday_drawdown_usd: float = 20_000.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for the RL training loop."""

    algorithm: str = "PPO"                    # PPO | TD3
    total_timesteps: int = 100_000
    n_envs: int = 1
    learning_rate: float = 3e-4
    n_steps: int = 2_048                      # PPO rollout length
    batch_size: int = 64
    policy_save_path: str = "models/execution_agent_policy"
    eval_episodes: int = 5                    # quick eval after training


# ---------------------------------------------------------------------------
# LLM / R-Quant system
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Settings for the AutoGen-based R-Quant pipeline."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None            # falls back to OPENAI_API_KEY env var
    use_mock: bool = True                    # set False to make real API calls
    max_conversation_turns: int = 10
    temperature: float = 0.2


# ---------------------------------------------------------------------------
# Backtest (used by R-Quant tools)
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Parameters for the lightweight backtest runner."""

    lookback_window: int = 20                # default signal lookback
    transaction_cost_bps: float = 5.0        # one-way cost in basis points
    initial_capital: float = 100_000.0


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityConfig:
    """Tracing / logging settings."""

    log_dir: str = "output/traces"
    log_level: str = "INFO"
    # Future: langsmith_project, langfuse_host, otel_endpoint, ...


# ---------------------------------------------------------------------------
# Top-level settings object
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Aggregate configuration for the entire system."""

    env: EnvConfig = field(default_factory=EnvConfig)
    local_limits: LocalRiskLimits = field(default_factory=LocalRiskLimits)
    global_limits: GlobalRiskLimits = field(default_factory=GlobalRiskLimits)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Orchestrator
    research_interval_episodes: int = 10    # run research cycle every N episodes
    output_dir: str = "output"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings, reading overrides from environment variables / .env file."""
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]

            load_dotenv()
        except ImportError:
            pass

        settings = cls()
        settings.llm.api_key = os.getenv("OPENAI_API_KEY")
        settings.llm.use_mock = os.getenv("USE_MOCK_LLM", "true").lower() != "false"
        return settings

    def output_path(self, *parts: str) -> Path:
        """Return an absolute path under output_dir, creating it if needed."""
        p = Path(self.output_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
