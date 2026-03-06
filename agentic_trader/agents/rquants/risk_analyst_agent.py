"""RiskAnalystAgent — interprets backtest results and flags concerns.

In mock mode: applies deterministic thresholds to decide pass/fail.
In LLM mode:  uses an AutoGen AssistantAgent to write a narrative risk report,
              then extracts a structured verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from agentic_trader.config.settings import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict container
# ---------------------------------------------------------------------------


@dataclass
class RiskVerdict:
    """Structured risk assessment of a backtested strategy."""

    passed: bool
    sharpe_ok: bool
    drawdown_ok: bool
    var_ok: bool
    narrative: str

    # Thresholds used (stored for audit trail)
    min_sharpe: float = 0.0
    max_drawdown_usd: float = 20_000.0
    max_var_usd: float = 5_000.0


# ---------------------------------------------------------------------------
# RiskAnalystAgent
# ---------------------------------------------------------------------------


class RiskAnalystAgent:
    """Evaluates backtest results against risk thresholds.

    Args:
        min_sharpe: Minimum acceptable Sharpe ratio.
        max_drawdown_usd: Maximum acceptable max drawdown in USD.
        max_var_usd: Maximum acceptable 95% VaR in USD.
        llm_config: LLM settings; if use_mock=True, no API calls are made.
    """

    def __init__(
        self,
        min_sharpe: float = 0.0,
        max_drawdown_usd: float = 20_000.0,
        max_var_usd: float = 5_000.0,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.min_sharpe = min_sharpe
        self.max_drawdown_usd = max_drawdown_usd
        self.max_var_usd = max_var_usd
        self.llm_config = llm_config or LLMConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, backtest_result: dict[str, Any], strategy_description: str = "") -> RiskVerdict:
        """Assess a backtest result dict and return a structured RiskVerdict.

        Args:
            backtest_result: Output from ``run_backtest``.
            strategy_description: Optional context for LLM narrative.
        """
        if not backtest_result.get("approved", False):
            return RiskVerdict(
                passed=False,
                sharpe_ok=False,
                drawdown_ok=False,
                var_ok=False,
                narrative=f"Backtest failed to run: {backtest_result.get('error', 'unknown error')}",
            )

        sharpe = backtest_result.get("sharpe_ratio", 0.0)
        drawdown = backtest_result.get("max_drawdown_usd", float("inf"))
        var = backtest_result.get("var_95_usd", float("inf"))

        sharpe_ok = sharpe >= self.min_sharpe
        drawdown_ok = drawdown <= self.max_drawdown_usd
        var_ok = var <= self.max_var_usd
        passed = sharpe_ok and drawdown_ok and var_ok

        if self.llm_config.use_mock:
            narrative = self._mock_narrative(backtest_result, sharpe_ok, drawdown_ok, var_ok)
        else:
            narrative = self._llm_narrative(backtest_result, strategy_description)

        verdict = RiskVerdict(
            passed=passed,
            sharpe_ok=sharpe_ok,
            drawdown_ok=drawdown_ok,
            var_ok=var_ok,
            narrative=narrative,
            min_sharpe=self.min_sharpe,
            max_drawdown_usd=self.max_drawdown_usd,
            max_var_usd=self.max_var_usd,
        )
        logger.info(
            "RiskAnalyst verdict: passed=%s | Sharpe=%.2f | MaxDD=$%.0f | VaR=$%.0f",
            passed,
            sharpe,
            drawdown,
            var,
        )
        return verdict

    # ------------------------------------------------------------------
    # Mock narrative
    # ------------------------------------------------------------------

    def _mock_narrative(
        self,
        result: dict[str, Any],
        sharpe_ok: bool,
        drawdown_ok: bool,
        var_ok: bool,
    ) -> str:
        lines = [
            f"Sharpe ratio: {result.get('sharpe_ratio', 0):.2f} "
            f"({'OK' if sharpe_ok else 'BELOW THRESHOLD'} vs min {self.min_sharpe:.2f})",
            f"Max drawdown: ${result.get('max_drawdown_usd', 0):,.0f} "
            f"({'OK' if drawdown_ok else 'EXCEEDS LIMIT'} vs max ${self.max_drawdown_usd:,.0f})",
            f"95% VaR: ${result.get('var_95_usd', 0):,.0f} "
            f"({'OK' if var_ok else 'EXCEEDS LIMIT'} vs max ${self.max_var_usd:,.0f})",
            f"Sortino ratio: {result.get('sortino_ratio', 0):.2f}",
            f"Total PnL: ${result.get('total_pnl_usd', 0):,.0f}",
        ]
        verdict_line = "RISK ASSESSMENT: PASS" if (sharpe_ok and drawdown_ok and var_ok) else "RISK ASSESSMENT: FAIL"
        return "\n".join([verdict_line, *lines])

    # ------------------------------------------------------------------
    # LLM narrative (AutoGen single-agent)
    # ------------------------------------------------------------------

    def _llm_narrative(self, result: dict[str, Any], strategy_description: str) -> str:
        """Request a narrative risk report from an LLM agent."""
        try:
            import autogen  # type: ignore[import-untyped]
        except ImportError:
            return self._mock_narrative(result, True, True, True)

        api_key = self.llm_config.api_key
        if not api_key:
            return self._mock_narrative(result, True, True, True)

        config_list = [{"model": self.llm_config.model, "api_key": api_key}]
        agent = autogen.AssistantAgent(
            name="risk_analyst",
            llm_config={"config_list": config_list, "temperature": 0.1},
            system_message=(
                "You are a risk analyst at a quant hedge fund. "
                "Write a concise (≤200 word) risk assessment of the provided backtest results. "
                "Highlight Sharpe, drawdown, VaR, and any tail-risk concerns. "
                "End with one line: RISK ASSESSMENT: PASS or RISK ASSESSMENT: FAIL."
            ),
        )
        user_proxy = autogen.UserProxyAgent(
            name="requester",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        user_proxy.initiate_chat(
            agent,
            message=(
                f"Strategy: {strategy_description}\n"
                f"Backtest results: {result}\n"
                f"Thresholds: min_sharpe={self.min_sharpe}, "
                f"max_drawdown=${self.max_drawdown_usd:,.0f}, "
                f"max_var=${self.max_var_usd:,.0f}"
            ),
        )
        # Extract the last message from the agent
        messages = user_proxy.chat_messages.get(agent, [])
        return messages[-1]["content"] if messages else "No narrative generated."
