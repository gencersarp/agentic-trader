"""DataCuratorAgent — ensures data is available before research begins.

Responsibilities:
  * Check whether requested data is on disk.
  * Generate / download it if not.
  * Return a standardised data handle to downstream agents.

In the PoC, this agent is a thin Python object (not an AutoGen ConversableAgent)
because it does not need LLM reasoning — pure data-engineering logic.  It is
designed to be upgraded to a real agent that can query data APIs if needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_trader.agents.rquants.tools import load_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data handle
# ---------------------------------------------------------------------------


@dataclass
class DataHandle:
    """Lightweight reference to a prepared dataset."""

    symbol: str
    start_date: str
    end_date: str
    n_bars: int
    prices_json: str   # JSON string — passed directly to run_backtest


# ---------------------------------------------------------------------------
# DataCuratorAgent
# ---------------------------------------------------------------------------


class DataCuratorAgent:
    """Prepares and validates datasets for downstream R-Quant agents.

    Args:
        data_dir: Directory to look for / save CSV files.
        cache: In-memory cache so repeated calls for the same symbol/range are free.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, DataHandle] = {}

    def prepare(self, symbol: str, start_date: str, end_date: str) -> DataHandle:
        """Return a DataHandle for the requested symbol / date range.

        Uses in-memory cache; falls back to ``load_data`` (which uses synthetic
        prices if no CSV is present).
        """
        key = f"{symbol}|{start_date}|{end_date}"
        if key in self._cache:
            logger.info("DataCurator: cache hit for %s", key)
            return self._cache[key]

        logger.info("DataCurator: loading %s %s→%s", symbol, start_date, end_date)
        result = load_data(symbol, start_date, end_date, data_dir=self.data_dir)

        handle = DataHandle(
            symbol=result["symbol"],
            start_date=result["start"],
            end_date=result["end"],
            n_bars=result["n_bars"],
            prices_json=result["prices_json"],
        )
        self._cache[key] = handle
        logger.info("DataCurator: prepared %d bars for %s", handle.n_bars, symbol)
        return handle

    def summary(self, handle: DataHandle) -> str:
        """Return a human-readable summary of the data handle."""
        return (
            f"Symbol: {handle.symbol} | Bars: {handle.n_bars} | "
            f"Range: {handle.start_date} – {handle.end_date}"
        )
