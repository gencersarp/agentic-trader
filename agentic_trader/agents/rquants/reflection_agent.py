"""ReflectionAgent — stores and retrieves Reflexion-style lessons.

After a poor backtest or trading episode, the agent:
  1. Summarises what went wrong (LLM or rule-based).
  2. Appends a structured "lesson" to a persistent JSON store.
  3. Exposes a retrieval interface so future research cycles can load relevant
     lessons as additional context (RAG-style).

Store format (output/reflections.json):
    [{"id": "...", "timestamp": "...", "strategy": "...", "lesson": "...",
      "metrics": {...}, "tags": [...]}, ...]
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agentic_trader.config.settings import LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = "output/reflections.json"


# ---------------------------------------------------------------------------
# Lesson dataclass
# ---------------------------------------------------------------------------


@dataclass
class Lesson:
    """A single stored lesson from a failed or underperforming strategy."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    strategy_description: str = ""
    lesson_text: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------


class ReflectionAgent:
    """Stores and retrieves Reflexion-style lessons.

    Args:
        store_path: Path to the JSON file used as the lesson store.
        llm_config: LLM settings; if use_mock=True, lessons are generated
            from a template without any LLM call.
    """

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.store_path = Path(store_path)
        self.llm_config = llm_config or LLMConfig()
        self._store: list[Lesson] = []
        self._load_store()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reflect(
        self,
        strategy_description: str,
        backtest_result: dict[str, Any],
        extra_context: str = "",
    ) -> Lesson:
        """Summarise a failure and append it to the lesson store.

        Args:
            strategy_description: The strategy that underperformed.
            backtest_result: Output from run_backtest.
            extra_context: Any additional information (e.g. regime at failure).

        Returns:
            The newly created Lesson (already persisted to disk).
        """
        if self.llm_config.use_mock:
            text = self._template_lesson(strategy_description, backtest_result, extra_context)
        else:
            text = self._llm_lesson(strategy_description, backtest_result, extra_context)

        tags = self._auto_tag(backtest_result)
        lesson = Lesson(
            strategy_description=strategy_description,
            lesson_text=text,
            metrics={
                k: backtest_result.get(k)
                for k in ("sharpe_ratio", "max_drawdown_usd", "var_95_usd", "total_pnl_usd")
                if k in backtest_result
            },
            tags=tags,
        )
        self._store.append(lesson)
        self._save_store()
        logger.info("ReflectionAgent: lesson %s stored (tags=%s)", lesson.id, tags)
        return lesson

    def retrieve(
        self,
        n: int = 3,
        tags: Optional[list[str]] = None,
    ) -> list[Lesson]:
        """Retrieve the most recent lessons, optionally filtered by tags.

        Args:
            n: Maximum number of lessons to return.
            tags: If provided, only lessons containing at least one of these
                  tags are included.
        """
        pool = self._store
        if tags:
            pool = [l for l in pool if any(t in l.tags for t in tags)]
        return pool[-n:]  # most recent first by insertion order

    def format_for_context(self, n: int = 3) -> str:
        """Return a formatted string of recent lessons for use as LLM context."""
        lessons = self.retrieve(n)
        if not lessons:
            return "No lessons stored yet."
        lines = [f"=== Past Lessons (most recent {len(lessons)}) ==="]
        for les in lessons:
            lines.append(
                f"\n[{les.timestamp[:10]}] {les.strategy_description}\n"
                f"  Lesson: {les.lesson_text}\n"
                f"  Metrics: {json.dumps(les.metrics)}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lesson generation
    # ------------------------------------------------------------------

    def _template_lesson(
        self,
        description: str,
        result: dict[str, Any],
        context: str,
    ) -> str:
        sharpe = result.get("sharpe_ratio", "N/A")
        dd = result.get("max_drawdown_usd", "N/A")
        pnl = result.get("total_pnl_usd", "N/A")
        lines = [
            f"Strategy '{description}' underperformed.",
            f"Sharpe={sharpe}, MaxDD=${dd}, TotalPnL=${pnl}.",
        ]
        if float(sharpe or 0) < 0:
            lines.append(
                "Negative Sharpe suggests the signal is contrarian to the actual regime. "
                "Consider inverting the signal or restricting to specific market conditions."
            )
        if float(dd or 0) > 10_000:
            lines.append(
                "Large drawdown detected.  The strategy may lack stop-loss logic or "
                "be over-leveraged.  Consider adding an inventory/VaR constraint."
            )
        if context:
            lines.append(f"Additional context: {context}")
        return " ".join(lines)

    def _llm_lesson(
        self,
        description: str,
        result: dict[str, Any],
        context: str,
    ) -> str:
        """Generate a lesson narrative via AutoGen."""
        try:
            import autogen  # type: ignore[import-untyped]
        except ImportError:
            return self._template_lesson(description, result, context)

        api_key = self.llm_config.api_key
        if not api_key:
            return self._template_lesson(description, result, context)

        config_list = [{"model": self.llm_config.model, "api_key": api_key}]
        agent = autogen.AssistantAgent(
            name="reflection_agent",
            llm_config={"config_list": config_list, "temperature": 0.2},
            system_message=(
                "You are a senior quant doing a post-mortem on a failed trading strategy. "
                "Write ≤80 words explaining what went wrong and one concrete improvement. "
                "Be specific — reference the actual metrics."
            ),
        )
        proxy = autogen.UserProxyAgent(
            name="requester", human_input_mode="NEVER",
            code_execution_config=False, max_consecutive_auto_reply=1,
        )
        proxy.initiate_chat(
            agent,
            message=(
                f"Strategy: {description}\n"
                f"Results: {result}\n"
                f"Context: {context}"
            ),
        )
        messages = proxy.chat_messages.get(agent, [])
        return messages[-1]["content"] if messages else self._template_lesson(description, result, context)

    # ------------------------------------------------------------------
    # Auto-tagging
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_tag(result: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        sharpe = result.get("sharpe_ratio", 0) or 0
        dd = result.get("max_drawdown_usd", 0) or 0
        if sharpe < 0:
            tags.append("negative_sharpe")
        if sharpe < 0.5:
            tags.append("low_sharpe")
        if dd > 15_000:
            tags.append("high_drawdown")
        if not tags:
            tags.append("general")
        return tags

    # ------------------------------------------------------------------
    # Store persistence
    # ------------------------------------------------------------------

    def _load_store(self) -> None:
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text())
                self._store = [Lesson(**item) for item in raw]
                logger.info("ReflectionAgent: loaded %d lessons from %s", len(self._store), self.store_path)
            except Exception as exc:
                logger.warning("ReflectionAgent: could not load store (%s), starting fresh.", exc)
                self._store = []
        else:
            self._store = []

    def _save_store(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps([asdict(l) for l in self._store], indent=2)
        )
