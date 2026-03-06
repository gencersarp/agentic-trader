"""StrategyGovernanceAgent — regime-aware policy registry.

Decides which ExecutionAgents are active given the current MarketRegime and
recent performance statistics.  All policies remain in the registry; the
governance agent only gates which ones are allowed to trade.

Design for extensibility:
  * `select_policies` uses rule-based logic for the PoC.
  * Replace or augment `_rule_based_selection` with a learned selector when
    more performance data is available.
  * The registry supports multiple named strategies so future ablation studies
    can compare them side-by-side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from agentic_trader.agents.execution_agent import ExecutionAgent
from agentic_trader.agents.regime_agent import MarketRegime
from agentic_trader.risk.risk_metrics import PerformanceStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------


@dataclass
class StrategyRecord:
    """Tracks one registered strategy's metadata and latest performance."""

    name: str
    agent: ExecutionAgent
    description: str = ""
    regimes_allowed: set[MarketRegime] = field(
        default_factory=lambda: {MarketRegime.CALM, MarketRegime.HIGH_VOL, MarketRegime.CRISIS}
    )
    latest_stats: Optional[PerformanceStats] = None
    active: bool = True                  # manually disable here if needed
    n_episodes: int = 0


# ---------------------------------------------------------------------------
# StrategyGovernanceAgent
# ---------------------------------------------------------------------------


class StrategyGovernanceAgent:
    """Maintains a registry of named strategies and selects active ones per step.

    Args:
        min_sharpe_to_activate: Strategies with a Sharpe below this are
            suspended during HIGH_VOL / CRISIS regimes (but kept in registry).
        drawdown_suspension_threshold_usd: Strategies with max_drawdown (in USD)
            above this are suspended until manual review.
        min_episodes_before_filter: Number of completed episodes required before
            performance-based filtering applies.  During warmup all strategies
            trade freely so early-episode noise does not cause immediate suspension.
    """

    def __init__(
        self,
        min_sharpe_to_activate: float = -1.5,
        drawdown_suspension_threshold_usd: float = 50_000.0,
        min_episodes_before_filter: int = 5,
    ):
        self.min_sharpe = min_sharpe_to_activate
        self.dd_threshold = drawdown_suspension_threshold_usd   # USD
        self.min_episodes_before_filter = min_episodes_before_filter
        self._registry: dict[str, StrategyRecord] = {}

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        agent: ExecutionAgent,
        description: str = "",
        regimes_allowed: Optional[set[MarketRegime]] = None,
    ) -> None:
        """Add or update a strategy in the registry."""
        allowed = regimes_allowed or {
            MarketRegime.CALM,
            MarketRegime.HIGH_VOL,
            MarketRegime.CRISIS,
        }
        self._registry[name] = StrategyRecord(
            name=name,
            agent=agent,
            description=description,
            regimes_allowed=allowed,
        )
        logger.info("Registered strategy '%s' (allowed regimes: %s)", name, allowed)

    def update_stats(self, name: str, stats: PerformanceStats) -> None:
        """Update performance statistics for a registered strategy."""
        if name in self._registry:
            self._registry[name].latest_stats = stats
            self._registry[name].n_episodes += 1

    def disable(self, name: str) -> None:
        """Manually disable a strategy (e.g., after a compliance flag)."""
        if name in self._registry:
            self._registry[name].active = False
            logger.warning("Strategy '%s' has been manually disabled.", name)

    def enable(self, name: str) -> None:
        if name in self._registry:
            self._registry[name].active = True

    @property
    def strategy_names(self) -> list[str]:
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Policy selection
    # ------------------------------------------------------------------

    def select_policies(
        self,
        regime: MarketRegime,
        performance_stats: Optional[dict[str, PerformanceStats]] = None,
    ) -> dict[str, ExecutionAgent]:
        """Return the subset of registered ExecutionAgents allowed to trade.

        Args:
            regime: Current market regime from RegimeAgent.
            performance_stats: Optional override dict — if provided, merges
                with stored stats (useful when Orchestrator has fresher data).

        Returns:
            Mapping of strategy name → ExecutionAgent for active strategies.
        """
        # Merge any externally provided stats
        if performance_stats:
            for name, stats in performance_stats.items():
                if name in self._registry:
                    self._registry[name].latest_stats = stats

        active: dict[str, ExecutionAgent] = {}
        for name, record in self._registry.items():
            verdict, reason = self._rule_based_selection(record, regime)
            if verdict:
                active[name] = record.agent
                logger.debug("Strategy '%s' ACTIVE in regime %s", name, regime.value)
            else:
                logger.debug(
                    "Strategy '%s' INACTIVE in regime %s: %s", name, regime.value, reason
                )

        if not active:
            logger.warning(
                "No active strategies in regime %s — market will not be traded.", regime.value
            )
        return active

    # ------------------------------------------------------------------
    # Rule-based selection logic
    # ------------------------------------------------------------------

    def _rule_based_selection(
        self, record: StrategyRecord, regime: MarketRegime
    ) -> tuple[bool, str]:
        """Return (allowed, reason_if_not).

        Rules (evaluated in order):
            1. Manually disabled               → rejected
            2. Regime not in allowed set       → rejected
            3. CRISIS: require Sharpe > -1     → suspended if bad
            4. HIGH_VOL: require Sharpe > min  → suspended if bad
            5. Max drawdown threshold breach   → suspended
            6. Otherwise                       → active
        """
        if not record.active:
            return False, "manually disabled"

        if regime not in record.regimes_allowed:
            return False, f"regime {regime.value} not in allowed set"

        stats = record.latest_stats
        # Skip performance-based filtering during the warmup period to prevent
        # early-episode noise from incorrectly suspending strategies.
        if stats is not None and record.n_episodes >= self.min_episodes_before_filter:
            if regime == MarketRegime.CRISIS and stats.sharpe_ratio < -1.5:
                return False, f"Sharpe {stats.sharpe_ratio:.2f} too low for CRISIS"

            if regime == MarketRegime.HIGH_VOL and stats.sharpe_ratio < self.min_sharpe:
                return False, f"Sharpe {stats.sharpe_ratio:.2f} below min {self.min_sharpe}"

            if stats.max_drawdown > self.dd_threshold:
                return (
                    False,
                    f"max_drawdown ${stats.max_drawdown:,.0f} exceeds ${self.dd_threshold:,.0f}",
                )

        return True, ""
