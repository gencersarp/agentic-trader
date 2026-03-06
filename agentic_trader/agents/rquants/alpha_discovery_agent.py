"""AlphaDiscoveryAgent — LLM-driven strategy proposal and backtest execution.

Two operating modes, selected via Settings.llm.use_mock:

MOCK mode (default — no API key required):
    Returns a pre-written mean-reversion signal definition and runs it through
    the real backtest engine.  Output is identical in structure to real mode.

REAL mode (set USE_MOCK_LLM=false + provide OPENAI_API_KEY):
    Spins up an AutoGen GroupChat with:
      * ``alpha_writer``   — proposes signal code given a strategy description.
      * ``alpha_critic``   — reviews the code for correctness and risk issues.
      * ``code_executor``  — executes the final code and returns results.
    The group chat terminates when ``code_executor`` returns a valid backtest
    result dict.
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

from agentic_trader.agents.rquants.data_curator_agent import DataHandle
from agentic_trader.agents.rquants.tools import run_backtest
from agentic_trader.config.settings import BacktestConfig, LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AlphaProposal:
    """Output of the AlphaDiscoveryAgent."""

    description: str
    strategy_code: str
    backtest_result: dict[str, Any]
    source: str = "mock"  # "mock" | "llm"


# ---------------------------------------------------------------------------
# Mock strategy library (used when use_mock=True)
# ---------------------------------------------------------------------------


_MEAN_REVERSION_CODE = textwrap.dedent(
    """\
    def generate_signals(prices):
        \"\"\"Z-score mean-reversion: fade moves > 1 std over a 20-day window.

        np and pd are pre-injected by the backtest sandbox — no import needed.
        \"\"\"
        window = 20
        returns = prices.pct_change()
        rolling_mean = returns.rolling(window).mean()
        rolling_std  = returns.rolling(window).std()
        zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        signal = pd.Series(0.0, index=prices.index)
        signal[zscore < -1.0] = 1.0   # buy on dip
        signal[zscore >  1.0] = -1.0  # sell on rally
        return signal.fillna(0.0)
    """
)

_MOMENTUM_CODE = textwrap.dedent(
    """\
    def generate_signals(prices):
        \"\"\"Simple 12-1 momentum: long if 12-month return > 0, else short.

        pd is pre-injected by the backtest sandbox — no import needed.
        \"\"\"
        slow = 252   # ~12 months
        fast = 21    # ~1 month (skip)
        long_ret  = prices.pct_change(slow)
        short_ret = prices.pct_change(fast)
        momentum  = long_ret - short_ret

        signal = pd.Series(0.0, index=prices.index)
        signal[momentum > 0] = 1.0
        signal[momentum < 0] = -1.0
        return signal.fillna(0.0)
    """
)

_MOCK_STRATEGIES: dict[str, tuple[str, str]] = {
    "mean_reversion": (
        "Z-score mean-reversion: fade daily return z-scores > 1 sigma over a 20-day window.",
        _MEAN_REVERSION_CODE,
    ),
    "momentum": (
        "12-1 cross-sectional momentum: long when 12-month return minus 1-month return > 0.",
        _MOMENTUM_CODE,
    ),
}


# ---------------------------------------------------------------------------
# AlphaDiscoveryAgent
# ---------------------------------------------------------------------------


class AlphaDiscoveryAgent:
    """Proposes trading signal code and backtests it.

    Args:
        llm_config: LLM settings (model, api_key, use_mock).
        backtest_config: Backtest parameters forwarded to ``run_backtest``.
        preferred_mock_strategy: Which mock strategy to use when use_mock=True.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        preferred_mock_strategy: str = "mean_reversion",
    ):
        self.llm_config = llm_config or LLMConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self._mock_strategy = preferred_mock_strategy

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def discover(self, description: str, data: DataHandle) -> AlphaProposal:
        """Generate a strategy and run it against the provided data.

        Args:
            description: Free-text strategy goal (e.g. "mean-reversion on equities").
            data: DataHandle from DataCuratorAgent.

        Returns:
            AlphaProposal with the strategy code and backtest results.
        """
        if self.llm_config.use_mock:
            return self._mock_discover(description, data)
        return self._llm_discover(description, data)

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _mock_discover(self, description: str, data: DataHandle) -> AlphaProposal:
        """Return a canned strategy that actually runs through the backtest engine."""
        desc_lower = description.lower()
        if "momentum" in desc_lower or "trend" in desc_lower:
            key = "momentum"
        else:
            key = self._mock_strategy

        strat_description, code = _MOCK_STRATEGIES.get(key, _MOCK_STRATEGIES["mean_reversion"])
        logger.info("AlphaDiscovery [mock]: running '%s' strategy", key)

        result = run_backtest(code, data.prices_json, self.backtest_config)
        return AlphaProposal(
            description=strat_description,
            strategy_code=code,
            backtest_result=result,
            source="mock",
        )

    # ------------------------------------------------------------------
    # LLM mode (AutoGen GroupChat)
    # ------------------------------------------------------------------

    def _llm_discover(self, description: str, data: DataHandle) -> AlphaProposal:
        """Use AutoGen multi-agent conversation to write and test a strategy."""
        try:
            import autogen  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyautogen is required for real LLM mode: pip install pyautogen"
            ) from exc

        api_key = self.llm_config.api_key
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set for real LLM mode. "
                "Set USE_MOCK_LLM=true to run without an API key."
            )

        config_list = [{"model": self.llm_config.model, "api_key": api_key}]
        llm_cfg = {
            "config_list": config_list,
            "temperature": self.llm_config.temperature,
            "cache_seed": None,
        }

        # ---- system prompts -------------------------------------------------
        writer_prompt = textwrap.dedent(
            f"""\
            You are a quantitative analyst.  Your task is to write Python code that
            implements a trading signal for the following strategy:

                "{description}"

            Requirements:
            - Define a single function: def generate_signals(prices: pd.Series) -> pd.Series
            - The function receives daily close prices (a pd.Series with DatetimeIndex).
            - It must return a pd.Series of the same length with values in {{-1, 0, +1}}.
            - Use only: numpy (as np), pandas (as pd), and standard Python.
            - Keep the code under 30 lines.
            - Output ONLY the Python code block, nothing else.
            """
        )

        critic_prompt = textwrap.dedent(
            """\
            You are a senior quant reviewer.  Review the proposed signal code for:
            1. Correctness — does it implement the stated strategy?
            2. Look-ahead bias — does it use future data?
            3. Risk — are there edge cases that could cause NaN, inf, or division by zero?
            If the code is acceptable, respond with: APPROVED
            If not, state the specific issue and suggest a minimal fix.
            """
        )

        executor_prompt = textwrap.dedent(
            """\
            You are a code executor.  When the critic says APPROVED, extract the
            generate_signals function from the conversation and return it verbatim
            inside a JSON block like:
            ```json
            {"strategy_code": "<escaped python code>"}
            ```
            Do not add any other text.
            """
        )

        # ---- agents ---------------------------------------------------------
        writer = autogen.AssistantAgent(
            name="alpha_writer", llm_config=llm_cfg, system_message=writer_prompt
        )
        critic = autogen.AssistantAgent(
            name="alpha_critic", llm_config=llm_cfg, system_message=critic_prompt
        )
        executor = autogen.AssistantAgent(
            name="code_executor", llm_config=llm_cfg, system_message=executor_prompt
        )
        user_proxy = autogen.UserProxyAgent(
            name="research_orchestrator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=self.llm_config.max_conversation_turns,
            code_execution_config=False,
            is_termination_msg=lambda m: "strategy_code" in m.get("content", ""),
        )

        groupchat = autogen.GroupChat(
            agents=[user_proxy, writer, critic, executor],
            messages=[],
            max_round=self.llm_config.max_conversation_turns,
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_cfg)

        user_proxy.initiate_chat(
            manager,
            message=f"Design a trading strategy: {description}\nData info: {data.n_bars} daily bars for {data.symbol}.",
        )

        # ---- extract code from conversation ---------------------------------
        code = self._extract_code_from_chat(groupchat.messages)
        result = run_backtest(code, data.prices_json, self.backtest_config)

        return AlphaProposal(
            description=description,
            strategy_code=code,
            backtest_result=result,
            source="llm",
        )

    @staticmethod
    def _extract_code_from_chat(messages: list[dict]) -> str:
        """Extract strategy_code from the last JSON block in the conversation."""
        import re

        for msg in reversed(messages):
            content = msg.get("content", "")
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return data.get("strategy_code", "")
                except json.JSONDecodeError:
                    continue
            # fallback: look for raw code block
            match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        raise ValueError("Could not extract strategy_code from AutoGen conversation.")
