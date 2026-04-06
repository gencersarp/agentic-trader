"""Risk metrics and the RiskEngine.

Implements historical-simulation and parametric (Gaussian) VaR, plus standard
performance statistics used throughout the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PerformanceStats:
    """Summary statistics for a single trading strategy or episode."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0          # positive USD amount (not a fraction)
    calmar_ratio: float = 0.0
    var_95: float = 0.0                # 1-period 95 % VaR (positive = loss)
    es_95: float = 0.0                 # Expected Shortfall at 95 %
    tail_ratio: float = 0.0            # 95th pct return / |5th pct return|
    n_periods: int = 0


@dataclass
class PortfolioState:
    """Snapshot of a live portfolio fed to the RiskGateway."""

    symbol: str
    inventory: float                   # shares (signed)
    mid_price: float
    cash: float
    initial_cash: float
    pnl_history: list[float] = field(default_factory=list)  # per-step P&L deltas

    @property
    def gross_notional(self) -> float:
        return abs(self.inventory) * self.mid_price

    @property
    def leverage(self) -> float:
        equity = self.cash + self.inventory * self.mid_price
        return self.gross_notional / max(abs(equity), 1.0)

    @property
    def intraday_pnl(self) -> float:
        return self.cash + self.inventory * self.mid_price - self.initial_cash

    @property
    def peak_pnl(self) -> float:
        if not self.pnl_history:
            return 0.0
        running = np.cumsum(self.pnl_history)
        return float(np.max(running)) if len(running) > 0 else 0.0

    @property
    def current_drawdown(self) -> float:
        """Absolute drawdown from peak (positive number)."""
        if not self.pnl_history:
            return 0.0
        running = np.cumsum(self.pnl_history)
        peak = np.maximum.accumulate(running)
        return float((peak - running)[-1])


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------


class RiskEngine:
    """Computes risk metrics from P&L time series.

    All VaR / ES figures are expressed as *positive* loss amounts in USD.
    """

    def __init__(self, confidence: float = 0.95):
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        self.confidence = confidence

    # ------------------------------------------------------------------
    # VaR methods
    # ------------------------------------------------------------------

    def historical_var(self, pnl_series: Sequence[float]) -> float:
        """Historical-simulation VaR: (1-confidence) left-tail quantile.

        Returns a *positive* number representing expected loss.
        """
        if len(pnl_series) < 10:
            return 0.0
        arr = np.array(pnl_series, dtype=float)
        quantile = 1.0 - self.confidence
        return float(-np.quantile(arr, quantile))

    def gaussian_var(self, pnl_series: Sequence[float]) -> float:
        """Parametric Gaussian VaR."""
        if len(pnl_series) < 5:
            return 0.0
        arr = np.array(pnl_series, dtype=float)
        mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
        z = stats.norm.ppf(1.0 - self.confidence)
        return float(-(mu + z * sigma))

    def expected_shortfall(self, pnl_series: Sequence[float]) -> float:
        """Historical ES (CVaR) — average of losses beyond VaR threshold."""
        if len(pnl_series) < 10:
            return 0.0
        arr = np.array(pnl_series, dtype=float)
        quantile = 1.0 - self.confidence
        threshold = np.quantile(arr, quantile)
        tail = arr[arr <= threshold]
        return float(-np.mean(tail)) if len(tail) > 0 else 0.0

    def estimate_var(self, portfolio: PortfolioState, hypothetical_order: dict) -> float:
        """Estimate 1-step VaR if a hypothetical order were to be executed.

        Uses parametric Gaussian VaR: recovers per-dollar price volatility
        from the PnL history and scales to the post-trade notional exposure.

        Args:
            portfolio: current portfolio snapshot
            hypothetical_order: dict with keys 'side' (+1/-1) and 'size' (shares)
        """
        size = hypothetical_order.get("size", 0)
        side = hypothetical_order.get("side", 0)
        new_inv = portfolio.inventory + side * size
        new_notional = abs(new_inv) * max(portfolio.mid_price, 0.01)
        z = stats.norm.ppf(self.confidence)  # ~1.65 for 95%

        history = list(portfolio.pnl_history)
        if len(history) < 10:
            # Not enough history — assume 1% per-step price volatility
            return float(new_notional * 0.01 * z)

        # Recover per-dollar volatility from PnL history.
        # PnL_t ~ inventory_t * delta_price_t, so std(PnL) ~ avg_|notional| * price_vol.
        arr = np.array(history, dtype=float)
        pnl_vol = float(np.std(arr, ddof=1))
        # Use current notional as the denominator (at least 1 share worth)
        avg_notional = max(portfolio.gross_notional, portfolio.mid_price)
        price_vol_frac = pnl_vol / avg_notional

        return float(max(0.0, new_notional * price_vol_frac * z))

    # ------------------------------------------------------------------
    # Performance statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stats(pnl_deltas: Sequence[float], periods_per_year: int = 98_280) -> PerformanceStats:
        """Compute full performance statistics from a series of per-period P&L deltas."""
        if len(pnl_deltas) == 0:
            return PerformanceStats()

        arr = np.array(pnl_deltas, dtype=float)
        n = len(arr)
        cumulative = float(np.sum(arr))
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # Sharpe (annualised, risk-free = 0 for simplicity)
        sharpe = (mu / sigma * np.sqrt(periods_per_year)) if sigma > 0 else 0.0

        # Sortino (downside deviation)
        downside = arr[arr < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = (mu / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

        # Max drawdown
        running = np.cumsum(arr)
        peak = np.maximum.accumulate(running)
        drawdowns = peak - running
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar
        calmar = (cumulative / max_dd) if max_dd > 0 else 0.0

        p95 = float(np.percentile(arr, 95))
        p05 = float(np.percentile(arr, 5))
        tail_ratio = p95 / max(abs(p05), 1e-9)

        engine = RiskEngine()
        return PerformanceStats(
            total_return=cumulative,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=max_dd,
            calmar_ratio=float(calmar),
            var_95=engine.historical_var(arr),
            es_95=engine.expected_shortfall(arr),
            tail_ratio=float(tail_ratio),
            n_periods=n,
        )

    @staticmethod
    def returns_from_prices(prices: Sequence[float]) -> np.ndarray:
        """Convert a price series to log-returns."""
        arr = np.array(prices, dtype=float)
        return np.diff(np.log(arr))
