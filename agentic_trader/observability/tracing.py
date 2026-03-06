"""Structured tracing and observability hooks.

Provides a minimal, self-contained tracing layer with a clean API.  The
implementation writes structured JSON lines to disk.  The API is designed so
that, in production, the body of each function can be replaced with calls to
LangSmith, Langfuse, or an OpenTelemetry exporter without changing any caller.

Trace hierarchy:
    TraceContext (one per high-level operation, e.g. one training episode)
      └── events (individual log_event calls)

Usage::

    ctx = start_trace("episode_42")
    log_event(ctx, "order_submitted", {"symbol": "AAPL", "size": 100, "side": "BUY"})
    log_event(ctx, "order_rejected", {"reason": "VAR_LIMIT", "estimated_var": 12500})
    end_trace(ctx, status="ok")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """A single event within a trace."""

    event_name: str
    payload: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    elapsed_ms: float = 0.0


@dataclass
class TraceContext:
    """Context object for a single traced operation."""

    trace_id: str
    trace_name: str
    start_time: float = field(default_factory=time.monotonic)
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1_000.0


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class Tracer:
    """Central tracing manager.

    Args:
        log_dir: Directory where JSONL trace files are written.
        log_level: Python logging level for console output.

    TODO (production upgrade):
        - Replace file-based sink with an OpenTelemetry OTLP exporter.
        - Add async flush via a background thread / asyncio task.
        - Integrate with LangSmith's `langsmith.Client` or Langfuse's SDK.
    """

    def __init__(self, log_dir: str = "output/traces", log_level: str = "INFO"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, TraceContext] = {}

        # Also mirror events to Python logger for developer convenience
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_trace(
        self, trace_name: str, metadata: Optional[dict[str, Any]] = None
    ) -> TraceContext:
        """Open a new trace context.

        Args:
            trace_name: Human-readable name (e.g. "episode_42", "research_cycle_3").
            metadata: Optional key-value metadata attached to the trace.

        Returns:
            A TraceContext object to pass to subsequent log_event / end_trace calls.
        """
        ctx = TraceContext(
            trace_id=str(uuid.uuid4())[:12],
            trace_name=trace_name,
            metadata=metadata or {},
        )
        self._active[ctx.trace_id] = ctx
        logger.debug("TRACE START | %s (%s)", trace_name, ctx.trace_id)
        return ctx

    def log_event(
        self, ctx: TraceContext, event_name: str, payload: dict[str, Any]
    ) -> None:
        """Append a structured event to the trace.

        Args:
            ctx: The active TraceContext.
            event_name: Short identifier (e.g. "order_submitted", "regime_changed").
            payload: Arbitrary key-value data — must be JSON-serialisable.
        """
        event = TraceEvent(
            event_name=event_name,
            payload=payload,
            elapsed_ms=ctx.elapsed_ms(),
        )
        ctx.events.append(event)
        logger.debug(
            "EVENT | %s | %s | %s",
            ctx.trace_name,
            event_name,
            json.dumps(payload, default=str),
        )

    def end_trace(self, ctx: TraceContext, status: str = "ok") -> None:
        """Close a trace and flush it to disk.

        Args:
            ctx: The TraceContext to close.
            status: "ok" | "error" | "cancelled" | custom string.
        """
        ctx.status = status
        self._active.pop(ctx.trace_id, None)
        self._flush(ctx)
        logger.debug(
            "TRACE END | %s (%s) | status=%s | %.0f ms",
            ctx.trace_name,
            ctx.trace_id,
            status,
            ctx.elapsed_ms(),
        )

    # ------------------------------------------------------------------
    # Convenience wrappers for common trading events
    # ------------------------------------------------------------------

    def log_order(
        self,
        ctx: TraceContext,
        event: str,
        symbol: str,
        side: str,
        size: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Shorthand for order-related events."""
        payload = {"symbol": symbol, "side": side, "size": size}
        if extra:
            payload.update(extra)
        self.log_event(ctx, event, payload)

    def log_regime(
        self, ctx: TraceContext, regime: str, features: Optional[dict[str, Any]] = None
    ) -> None:
        payload: dict[str, Any] = {"regime": regime}
        if features:
            payload["features"] = features
        self.log_event(ctx, "regime_classified", payload)

    def log_episode_summary(
        self,
        ctx: TraceContext,
        episode: int,
        total_pnl: float,
        total_reward: float,
        n_orders: int,
        n_rejections: int,
        regime_counts: dict[str, int],
    ) -> None:
        self.log_event(
            ctx,
            "episode_summary",
            {
                "episode": episode,
                "total_pnl": round(total_pnl, 2),
                "total_reward": round(total_reward, 4),
                "n_orders": n_orders,
                "n_rejections": n_rejections,
                "regime_counts": regime_counts,
            },
        )

    # ------------------------------------------------------------------
    # File sink
    # ------------------------------------------------------------------

    def _flush(self, ctx: TraceContext) -> None:
        """Write a complete trace to a JSONL file."""
        out_path = self._log_dir / f"{ctx.trace_name}.jsonl"
        record = {
            "trace_id": ctx.trace_id,
            "trace_name": ctx.trace_name,
            "status": ctx.status,
            "duration_ms": round(ctx.elapsed_ms(), 2),
            "metadata": ctx.metadata,
            "events": [asdict(e) for e in ctx.events],
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Module-level default tracer (convenience singleton)
# ---------------------------------------------------------------------------

_default_tracer: Optional[Tracer] = None


def get_tracer(log_dir: str = "output/traces") -> Tracer:
    """Return (or lazily create) the module-level default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer(log_dir=log_dir)
    return _default_tracer


# ---------------------------------------------------------------------------
# Free functions that mirror Tracer methods (for callers that prefer them)
# ---------------------------------------------------------------------------


def start_trace(trace_name: str, metadata: Optional[dict[str, Any]] = None) -> TraceContext:
    return get_tracer().start_trace(trace_name, metadata)


def log_event(ctx: TraceContext, event_name: str, payload: dict[str, Any]) -> None:
    get_tracer().log_event(ctx, event_name, payload)


def end_trace(ctx: TraceContext, status: str = "ok") -> None:
    get_tracer().end_trace(ctx, status)
