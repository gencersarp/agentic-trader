"""Shared tool functions available to all R-Quant agents.

These are plain Python callables registered with the AutoGen agents as
function tools.  They are intentionally *pure* (no side effects beyond disk
writes) and type-annotated so that AutoGen can generate accurate JSON schemas.

Design note: keeping tools separate from agents makes them easy to unit-test
independently and trivial to expose via a REST API in production.
"""

from __future__ import annotations

import io
import json
import logging
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_trader.config.settings import BacktestConfig
from agentic_trader.risk.risk_metrics import RiskEngine, PerformanceStats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data tool
# ---------------------------------------------------------------------------


def load_data(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data",
) -> dict[str, Any]:
    """Load historical daily close prices for a symbol.

    Looks for ``{data_dir}/{symbol}.csv``.  If the file does not exist, falls
    back to generating synthetic GBM prices so the PoC can run without real data.

    Returns a dict with keys:
        ``symbol``, ``start``, ``end``, ``n_bars``, ``prices_json``
        (a JSON-serialised list of (date, close) pairs).
    """
    csv_path = Path(data_dir) / f"{symbol}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        prices = df["close"].loc[start_date:end_date]
    else:
        logger.warning(
            "No data file found for %s at %s — generating synthetic prices.", symbol, csv_path
        )
        prices = _synthetic_prices(symbol, start_date, end_date)

    prices = prices.dropna()
    result = {
        "symbol": symbol,
        "start": str(prices.index.min().date()) if len(prices) else start_date,
        "end": str(prices.index.max().date()) if len(prices) else end_date,
        "n_bars": len(prices),
        "prices_json": prices.reset_index().rename(columns={"date": "date"}).to_json(orient="records"),
    }
    return result


def _synthetic_prices(symbol: str, start: str, end: str, annual_vol: float = 0.25) -> pd.Series:
    """Generate synthetic GBM daily close prices."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**31))
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    dt = 1 / 252
    returns = rng.normal((0.05 - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


# ---------------------------------------------------------------------------
# Backtest tool
# ---------------------------------------------------------------------------


def run_backtest(
    strategy_code: str,
    prices_json: str,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Execute a strategy backtest from LLM-generated Python code.

    The ``strategy_code`` must define a function::

        def generate_signals(prices: pd.Series) -> pd.Series:
            # returns: pd.Series of float in {-1, 0, +1}
            # np (numpy) and pd (pandas) are pre-injected — do NOT import them.

    The backtest runner executes the code in a *restricted* namespace that
    pre-injects ``np`` and ``pd``.  Do not add ``import`` statements inside
    ``generate_signals``; they will raise ImportError in the sandbox.

    Args:
        strategy_code: Python source string containing ``generate_signals``.
        prices_json: JSON string of {date, close} records (from ``load_data``).
        config: Backtest parameters (lookback, costs, capital).

    Returns:
        Dict with performance metrics and a short summary string.
    """
    cfg = config or BacktestConfig()

    # --- reconstruct prices --------------------------------------------------
    try:
        records = json.loads(prices_json)
        df = pd.DataFrame(records)
        date_col = "date" if "date" in df.columns else df.columns[0]
        close_col = "close" if "close" in df.columns else df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        prices = df[close_col].astype(float)
    except Exception as exc:
        return {"error": f"Failed to parse prices_json: {exc}", "approved": False}

    # --- execute strategy code in sandbox ------------------------------------
    allowed_globals: dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "max": max, "min": min, "len": len,
            "range": range, "zip": zip, "enumerate": enumerate,
            "print": print, "round": round,
        },
        "pd": pd,
        "np": np,
    }
    local_ns: dict[str, Any] = {}
    try:
        exec(strategy_code, allowed_globals, local_ns)  # noqa: S102
    except Exception as exc:
        return {"error": f"Strategy code execution failed: {exc}", "approved": False}

    if "generate_signals" not in local_ns:
        return {"error": "strategy_code must define 'generate_signals(prices)'", "approved": False}

    # --- run signal generation -----------------------------------------------
    try:
        signals: pd.Series = local_ns["generate_signals"](prices)
        signals = signals.reindex(prices.index).fillna(0).clip(-1, 1)
    except Exception as exc:
        tb = traceback.format_exc()
        return {"error": f"generate_signals raised: {exc}\n{tb}", "approved": False}

    # --- vectorised P&L engine -----------------------------------------------
    position = signals.shift(1).fillna(0)   # execute on next open (simplified)
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = position * daily_returns

    # transaction costs: cost_bps per side, applied on position changes
    trades = position.diff().abs().fillna(0)
    cost_rate = cfg.transaction_cost_bps / 10_000.0
    costs = trades * cost_rate

    net_returns = strategy_returns - costs
    pnl_series = net_returns * cfg.initial_capital

    # --- compute stats -------------------------------------------------------
    engine = RiskEngine()
    stats = RiskEngine.compute_stats(pnl_series.tolist())

    return {
        "sharpe_ratio": round(stats.sharpe_ratio, 4),
        "sortino_ratio": round(stats.sortino_ratio, 4),
        "max_drawdown_usd": round(stats.max_drawdown, 2),
        "total_pnl_usd": round(stats.total_return, 2),
        "var_95_usd": round(stats.var_95, 2),
        "es_95_usd": round(stats.es_95, 2),
        "n_periods": stats.n_periods,
        "summary": (
            f"Sharpe={stats.sharpe_ratio:.2f}, Sortino={stats.sortino_ratio:.2f}, "
            f"MaxDD=${stats.max_drawdown:,.0f}, Total PnL=${stats.total_return:,.0f}"
        ),
        "approved": True,
    }


# ---------------------------------------------------------------------------
# Risk metrics tool
# ---------------------------------------------------------------------------


def compute_risk_metrics(pnl_list: list[float]) -> dict[str, float]:
    """Compute risk metrics from a list of P&L deltas.

    Args:
        pnl_list: per-period P&L deltas in USD.

    Returns:
        Dict with Sharpe, Sortino, max_drawdown, var_95, es_95.
    """
    stats = RiskEngine.compute_stats(pnl_list)
    return {
        "sharpe_ratio": round(stats.sharpe_ratio, 4),
        "sortino_ratio": round(stats.sortino_ratio, 4),
        "max_drawdown_usd": round(stats.max_drawdown, 2),
        "total_pnl_usd": round(stats.total_return, 2),
        "var_95_usd": round(stats.var_95, 2),
        "es_95_usd": round(stats.es_95, 2),
    }
