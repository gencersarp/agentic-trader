"""Orchestrator — wires every layer of the system together.

Implements two main loops:

training_loop()
    Runs N episodes in the simulation, routing each agent's proposed action
    through the RiskGateway before handing it to the environment.  Tracks P&L
    and performance statistics per strategy, updates StrategyGovernanceAgent,
    and periodically triggers a research cycle.

research_cycle()
    Invokes the R-Quant multi-agent pipeline to propose and evaluate a new
    strategy.  Successful proposals are registered as candidates in the
    StrategyGovernanceAgent.

Both loops are fully instrumented via the Tracer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agentic_trader.agents.execution_agent import ExecutionAgent
from agentic_trader.agents.regime_agent import MarketRegime, RegimeAgent
from agentic_trader.agents.rquants.alpha_discovery_agent import AlphaDiscoveryAgent
from agentic_trader.agents.rquants.compliance_agent import ComplianceAgent
from agentic_trader.agents.rquants.data_curator_agent import DataCuratorAgent
from agentic_trader.agents.rquants.reflection_agent import ReflectionAgent
from agentic_trader.agents.rquants.risk_analyst_agent import RiskAnalystAgent
from agentic_trader.agents.strategy_governance_agent import StrategyGovernanceAgent
from agentic_trader.config.settings import Settings
from agentic_trader.env.abides_env import TradingEnv
from agentic_trader.observability.tracing import Tracer, get_tracer
from agentic_trader.risk.risk_gateway import Order, OrderSide, RiskGateway
from agentic_trader.risk.risk_metrics import PerformanceStats, PortfolioState, RiskEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Per-episode summary collected by the Orchestrator."""

    episode: int
    total_pnl: float = 0.0
    total_reward: float = 0.0
    n_steps: int = 0
    n_orders_submitted: int = 0
    n_orders_rejected: int = 0
    regime_counts: dict[str, int] = field(default_factory=dict)
    performance: Optional[PerformanceStats] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Central coordinator for the multi-agent trading system.

    Args:
        env: The simulation environment.
        execution_agents: Dict mapping strategy name → ExecutionAgent.
        risk_gateway: Pre-trade risk filter.
        regime_agent: Market regime classifier.
        strategy_agent: Policy registry and regime-aware selector.
        settings: Global configuration.
        tracer: Optional Tracer instance (creates default if None).
    """

    def __init__(
        self,
        env: TradingEnv,
        execution_agents: dict[str, ExecutionAgent],
        risk_gateway: RiskGateway,
        regime_agent: RegimeAgent,
        strategy_agent: StrategyGovernanceAgent,
        settings: Optional[Settings] = None,
        tracer: Optional[Tracer] = None,
    ):
        self.env = env
        self.risk_gateway = risk_gateway
        self.regime_agent = regime_agent
        self.strategy_agent = strategy_agent
        self.settings = settings or Settings()
        self.tracer = tracer or get_tracer(self.settings.observability.log_dir)

        # Register all execution agents
        for name, agent in execution_agents.items():
            self.strategy_agent.register(name, agent)

        # R-Quant pipeline (lazy initialised on first research_cycle call)
        self._data_curator: Optional[DataCuratorAgent] = None
        self._alpha_discovery: Optional[AlphaDiscoveryAgent] = None
        self._risk_analyst: Optional[RiskAnalystAgent] = None
        self._compliance: Optional[ComplianceAgent] = None
        self._reflection: Optional[ReflectionAgent] = None

        # History
        self._episode_history: list[EpisodeStats] = []
        self._pnl_buf: list[float] = []   # per-step P&L deltas for risk calcs
        self._risk_engine = RiskEngine()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def training_loop(self, n_episodes: int = 10) -> list[EpisodeStats]:
        """Run n_episodes of simulation and return per-episode statistics.

        Args:
            n_episodes: Number of complete episodes to simulate.

        Returns:
            List of EpisodeStats, one per episode.
        """
        logger.info("Orchestrator: starting training loop (%d episodes)", n_episodes)

        for ep in range(n_episodes):
            stats = self._run_episode(ep)
            self._episode_history.append(stats)

            # Update governance agent with latest performance
            perf = self._risk_engine.compute_stats(self._pnl_buf[-500:])
            for name in self.strategy_agent.strategy_names:
                self.strategy_agent.update_stats(name, perf)

            # Trigger research cycle periodically
            if (ep + 1) % self.settings.research_interval_episodes == 0:
                logger.info("Orchestrator: triggering research cycle after episode %d", ep)
                self.research_cycle()

            dom_regime = max(stats.regime_counts, key=stats.regime_counts.get, default="N/A")
            logger.info(
                "Episode %3d | PnL=%+.0f | reward=%.2f | orders=%d | rejected=%d | regime=%s",
                ep,
                stats.total_pnl,
                stats.total_reward,
                stats.n_orders_submitted,
                stats.n_orders_rejected,
                dom_regime,
            )

        logger.info("Orchestrator: training loop complete.")
        return self._episode_history

    def _run_episode(self, episode_idx: int) -> EpisodeStats:
        """Execute one full episode and return its statistics."""
        obs, _ = self.env.reset()
        self.regime_agent.reset()
        for agent in self.strategy_agent._registry.values():
            agent.agent.reset()
        self.risk_gateway.reset_counts()

        stats = EpisodeStats(episode=episode_idx)
        pnl_deltas: list[float] = []
        prev_pnl = 0.0

        trace = self.tracer.start_trace(
            f"episode_{episode_idx}",
            metadata={"episode": episode_idx, "symbol": self.env.config.symbol},
        )

        done = False
        while not done:
            # --- regime classification ---
            regime_features = self.env.get_regime_features()
            regime = self.regime_agent.classify(regime_features)
            regime_name = regime.value
            stats.regime_counts[regime_name] = stats.regime_counts.get(regime_name, 0) + 1

            # --- select active policies ---
            active_agents = self.strategy_agent.select_policies(regime)
            if not active_agents:
                # No active policies — advance environment with no-op
                obs, reward, terminated, truncated, info = self.env.step(np.array([0.0]))
                done = terminated or truncated
                stats.n_steps += 1
                stats.total_reward += float(reward)
                continue

            # --- collect and validate actions ---
            chosen_action = np.array([0.0], dtype=np.float32)
            for name, agent in active_agents.items():
                raw_action, order = agent.act(obs)

                if order is None:
                    continue   # dead-band — no trade

                stats.n_orders_submitted += 1
                portfolio = self._build_portfolio_state(info_prev={"pnl": prev_pnl})
                validation = self.risk_gateway.validate_order(order, portfolio)

                if validation.approved:
                    chosen_action = raw_action
                    self.tracer.log_order(
                        trace, "order_approved",
                        order.symbol, order.side.name, order.size,
                        {"agent": name, "regime": regime_name},
                    )
                else:
                    stats.n_orders_rejected += 1
                    self.tracer.log_order(
                        trace, "order_rejected",
                        order.symbol, order.side.name, order.size,
                        {"reason": validation.reason.value if validation.reason else "UNKNOWN",
                         "agent": name},
                    )

            # --- step environment ---
            obs, reward, terminated, truncated, info = self.env.step(chosen_action)
            done = terminated or truncated

            # --- accounting ---
            current_pnl = info.get("pnl", 0.0)
            delta = current_pnl - prev_pnl
            pnl_deltas.append(delta)
            self._pnl_buf.append(delta)
            prev_pnl = current_pnl

            stats.n_steps += 1
            stats.total_reward += float(reward)
            stats.total_pnl = current_pnl

        # --- end-of-episode stats ---
        stats.performance = self._risk_engine.compute_stats(pnl_deltas)
        self.tracer.log_episode_summary(
            trace,
            episode=episode_idx,
            total_pnl=stats.total_pnl,
            total_reward=stats.total_reward,
            n_orders=stats.n_orders_submitted,
            n_rejections=stats.n_orders_rejected,
            regime_counts=stats.regime_counts,
        )
        self.tracer.end_trace(trace, status="ok")
        return stats

    def _build_portfolio_state(self, info_prev: dict) -> PortfolioState:
        """Construct a PortfolioState from current env state for risk checks."""
        return PortfolioState(
            symbol=self.env.config.symbol,
            inventory=self.env.current_inventory,
            mid_price=self.env._mid,
            cash=self.env._cash,
            initial_cash=self.env.config.initial_cash,
            pnl_history=list(self._pnl_buf[-100:]),
        )

    # ------------------------------------------------------------------
    # Research cycle
    # ------------------------------------------------------------------

    def research_cycle(
        self,
        strategy_description: str = "mean-reversion on daily close prices",
        symbol: str = "SYNTHETIC",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ) -> Optional[dict]:
        """Run one complete R-Quant research cycle.

        Pipeline:
            DataCurator → AlphaDiscovery → RiskAnalyst → Compliance → Reflection

        Returns:
            A dict summarising the research outcome, or None on failure.
        """
        logger.info("Orchestrator: research cycle | '%s'", strategy_description)
        trace = self.tracer.start_trace("research_cycle", {"strategy": strategy_description})

        try:
            self._init_rquants()

            # 1. Prepare data
            self.tracer.log_event(trace, "data_curation_start", {"symbol": symbol})
            data = self._data_curator.prepare(symbol, start_date, end_date)  # type: ignore[union-attr]
            self.tracer.log_event(trace, "data_curation_done", {"n_bars": data.n_bars})

            # 2. Alpha discovery + backtest
            self.tracer.log_event(trace, "alpha_discovery_start", {})
            proposal = self._alpha_discovery.discover(strategy_description, data)  # type: ignore[union-attr]
            self.tracer.log_event(trace, "alpha_discovery_done", proposal.backtest_result)

            # 3. Risk assessment
            self.tracer.log_event(trace, "risk_analysis_start", {})
            risk_verdict = self._risk_analyst.evaluate(  # type: ignore[union-attr]
                proposal.backtest_result, strategy_description
            )
            self.tracer.log_event(
                trace, "risk_analysis_done", {"passed": risk_verdict.passed}
            )

            # 4. Compliance
            self.tracer.log_event(trace, "compliance_check_start", {})
            compliance_result = self._compliance.check(  # type: ignore[union-attr]
                proposal.description, proposal.strategy_code
            )
            self.tracer.log_event(
                trace, "compliance_check_done", {"passed": compliance_result.passed}
            )

            # 5. Reflection (always run regardless of outcome)
            overall_pass = risk_verdict.passed and compliance_result.passed
            if not overall_pass:
                lesson = self._reflection.reflect(  # type: ignore[union-attr]
                    proposal.description,
                    proposal.backtest_result,
                    extra_context=f"risk_passed={risk_verdict.passed}, "
                    f"compliance_passed={compliance_result.passed}",
                )
                self.tracer.log_event(trace, "lesson_stored", {"lesson_id": lesson.id})

            outcome = {
                "strategy_description": proposal.description,
                "source": proposal.source,
                "backtest": proposal.backtest_result,
                "risk_passed": risk_verdict.passed,
                "risk_narrative": risk_verdict.narrative,
                "compliance_passed": compliance_result.passed,
                "compliance_narrative": compliance_result.narrative,
                "overall_approved": overall_pass,
            }

            if overall_pass:
                logger.info(
                    "Research cycle: strategy APPROVED | Sharpe=%.2f",
                    proposal.backtest_result.get("sharpe_ratio", 0),
                )
            else:
                logger.warning("Research cycle: strategy REJECTED — see output/reflections.json")

            self.tracer.end_trace(trace, status="ok" if overall_pass else "rejected")
            return outcome

        except Exception as exc:
            logger.exception("Research cycle failed: %s", exc)
            self.tracer.end_trace(trace, status="error")
            return None

    def _init_rquants(self) -> None:
        """Lazily initialise R-Quant agents on first call."""
        if self._data_curator is not None:
            return
        cfg = self.settings
        self._data_curator = DataCuratorAgent()
        self._alpha_discovery = AlphaDiscoveryAgent(
            llm_config=cfg.llm, backtest_config=cfg.backtest
        )
        self._risk_analyst = RiskAnalystAgent(llm_config=cfg.llm)
        self._compliance = ComplianceAgent(llm_config=cfg.llm)
        self._reflection = ReflectionAgent(
            store_path=cfg.output_path("reflections.json"),
            llm_config=cfg.llm,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        """Return aggregate statistics over all completed episodes."""
        if not self._episode_history:
            return {}
        pnls = [e.total_pnl for e in self._episode_history]
        rewards = [e.total_reward for e in self._episode_history]
        rejections = [e.n_orders_rejected for e in self._episode_history]
        orders = [e.n_orders_submitted for e in self._episode_history]
        return {
            "n_episodes": len(self._episode_history),
            "mean_pnl": float(np.mean(pnls)),
            "std_pnl": float(np.std(pnls)),
            "total_pnl": float(np.sum(pnls)),
            "mean_reward": float(np.mean(rewards)),
            "total_orders": int(np.sum(orders)),
            "total_rejections": int(np.sum(rejections)),
            "rejection_rate": float(np.sum(rejections) / max(np.sum(orders), 1)),
        }
