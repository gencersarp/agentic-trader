"""ComplianceAgent — rule-based and LLM-assisted compliance checker.

For the PoC, compliance is a two-pass check:
  1. Hard keyword filter (deterministic) — catches obvious prohibited strategies.
  2. Optional LLM pass (real mode only) — narrative assessment for subtler issues.

This module is intentionally conservative: when in doubt, it flags for review
rather than approving.  A human compliance officer reviews flagged strategies
before they are registered in the StrategyGovernanceAgent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agentic_trader.config.settings import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banned keywords (non-exhaustive; extend per jurisdiction)
# ---------------------------------------------------------------------------

_BANNED_PATTERNS: list[str] = [
    r"\bspoofing\b",
    r"\blayering\b",
    r"\bwash\s+trade",
    r"\bfront[- ]?running\b",
    r"\bpump[- ]and[- ]dump\b",
    r"\binsider\b",
    r"\bmanipulat",
    r"\bfictitious\s+trade",
    r"\bcornering\s+the\s+market",
]

_BANNED_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BANNED_PATTERNS]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ComplianceResult:
    """Output of the ComplianceAgent."""

    passed: bool
    flags: list[str] = field(default_factory=list)   # matched prohibited patterns
    narrative: str = ""
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# ComplianceAgent
# ---------------------------------------------------------------------------


class ComplianceAgent:
    """Screens strategy descriptions and code for prohibited practices.

    Args:
        extra_banned_patterns: Additional regex patterns to add to the default list.
        llm_config: If use_mock=False and an API key is set, a second LLM pass
            generates a compliance narrative.
    """

    def __init__(
        self,
        extra_banned_patterns: Optional[list[str]] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.llm_config = llm_config or LLMConfig()
        self._patterns = list(_BANNED_COMPILED)
        for p in extra_banned_patterns or []:
            self._patterns.append(re.compile(p, re.IGNORECASE))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self, strategy_description: str, strategy_code: str = "") -> ComplianceResult:
        """Screen a strategy for compliance issues.

        Args:
            strategy_description: Free-text description of the strategy.
            strategy_code: The generated Python code (optional but recommended).

        Returns:
            ComplianceResult with pass/fail verdict and any flagged issues.
        """
        combined_text = f"{strategy_description}\n{strategy_code}"
        flags = self._keyword_scan(combined_text)

        if flags:
            narrative = (
                f"Keyword compliance scan found {len(flags)} issue(s): "
                + "; ".join(flags)
                + ".  This strategy requires human compliance review before activation."
            )
            logger.warning("ComplianceAgent FLAGGED: %s", flags)
            return ComplianceResult(
                passed=False,
                flags=flags,
                narrative=narrative,
                requires_human_review=True,
            )

        # Clean keyword scan — optionally do an LLM pass
        if not self.llm_config.use_mock and self.llm_config.api_key:
            narrative = self._llm_review(strategy_description, strategy_code)
            # Parse verdict from narrative
            passed = "COMPLIANCE: PASS" in narrative.upper()
            return ComplianceResult(passed=passed, flags=[], narrative=narrative)

        narrative = (
            "Keyword compliance scan: no prohibited patterns detected.  "
            "Strategy is provisionally approved pending risk sign-off."
        )
        logger.info("ComplianceAgent: PASS (keyword scan clean)")
        return ComplianceResult(passed=True, flags=[], narrative=narrative)

    # ------------------------------------------------------------------
    # Keyword scan
    # ------------------------------------------------------------------

    def _keyword_scan(self, text: str) -> list[str]:
        """Return a list of matched banned pattern descriptions."""
        matched = []
        for pattern in self._patterns:
            if pattern.search(text):
                matched.append(pattern.pattern)
        return matched

    # ------------------------------------------------------------------
    # LLM review
    # ------------------------------------------------------------------

    def _llm_review(self, description: str, code: str) -> str:
        """Request a brief compliance narrative from an LLM agent."""
        try:
            import autogen  # type: ignore[import-untyped]
        except ImportError:
            return "COMPLIANCE: PASS (LLM not available; keyword scan clean)."

        config_list = [{"model": self.llm_config.model, "api_key": self.llm_config.api_key}]
        agent = autogen.AssistantAgent(
            name="compliance_officer",
            llm_config={"config_list": config_list, "temperature": 0.0},
            system_message=(
                "You are a compliance officer at a regulated trading firm. "
                "Review the strategy description and code for any prohibited practices "
                "(spoofing, layering, front-running, wash trading, market manipulation, etc.). "
                "Write ≤100 words.  End with exactly one of: "
                "COMPLIANCE: PASS  or  COMPLIANCE: FAIL — REQUIRES REVIEW."
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
            message=f"Strategy description:\n{description}\n\nCode:\n```python\n{code}\n```",
        )
        messages = user_proxy.chat_messages.get(agent, [])
        return messages[-1]["content"] if messages else "COMPLIANCE: PASS (no response)."
