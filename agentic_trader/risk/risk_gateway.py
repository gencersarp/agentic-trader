"""Deterministic risk gateway — the hard outer shell around every agent.

Design principle: the gateway never trusts agent outputs.  It treats every
proposed order as untrusted and independently re-validates it against global
risk limits.  Even if an LLM hallucinates an order or a bug in an RL policy
produces an extreme action, the gateway catches it before it reaches the
exchange.

This is intentionally kept *deterministic* — no ML, no randomness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from agentic_trader.config.settings import GlobalRiskLimits
from agentic_trader.risk.risk_metrics import PortfolioState, RiskEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order representation
# ---------------------------------------------------------------------------


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


@dataclass
class Order:
    """A proposed order before risk validation."""

    symbol: str
    side: OrderSide
    size: int                   # shares (always positive)
    limit_price: Optional[float] = None   # None = market order
    agent_id: str = "unknown"

    @property
    def signed_size(self) -> int:
        return self.size if self.side == OrderSide.BUY else -self.size


# ---------------------------------------------------------------------------
# Rejection reason
# ---------------------------------------------------------------------------


class RejectionReason(str, Enum):
    POSITION_LIMIT = "POSITION_LIMIT"
    NOTIONAL_LIMIT = "NOTIONAL_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    VAR_LIMIT = "VAR_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    ZERO_SIZE = "ZERO_SIZE"


@dataclass
class ValidationResult:
    approved: bool
    reason: Optional[RejectionReason] = None
    detail: str = ""


# ---------------------------------------------------------------------------
# RiskGateway
# ---------------------------------------------------------------------------


class RiskGateway:
    """Deterministic pre-trade risk filter.

    All checks run in order; the first failure short-circuits and the order
    is rejected.  Approved orders are passed to the environment unchanged.

    Usage::

        gateway = RiskGateway(global_limits, risk_engine)
        result = gateway.validate_order(order, portfolio_state)
        if result.approved:
            env.step(order_to_action(order))
    """

    def __init__(self, global_limits: GlobalRiskLimits, risk_engine: Optional[RiskEngine] = None):
        self.limits = global_limits
        self.risk_engine = risk_engine or RiskEngine()
        self._rejection_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_order(self, order: Order, portfolio: PortfolioState) -> ValidationResult:
        """Run all pre-trade risk checks.

        Returns a ValidationResult.  If approved is False, reason and detail
        contain structured information suitable for logging / compliance audit.
        """
        for check in (
            self._check_zero_size,
            self._check_position_limits,
            self._check_notional_limit,
            self._check_leverage_limit,
            self._check_var_limit,
            self._check_drawdown_limit,
        ):
            result = check(order, portfolio)
            if not result.approved:
                self._record_rejection(order, result)
                return result

        return ValidationResult(approved=True)

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Cumulative rejection counts keyed by RejectionReason value."""
        return dict(self._rejection_counts)

    def reset_counts(self) -> None:
        self._rejection_counts.clear()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_zero_size(self, order: Order, portfolio: PortfolioState) -> ValidationResult:
        if order.size <= 0:
            return ValidationResult(
                approved=False,
                reason=RejectionReason.ZERO_SIZE,
                detail="Order size must be positive.",
            )
        return ValidationResult(approved=True)

    def _check_position_limits(self, order: Order, portfolio: PortfolioState) -> ValidationResult:
        """Ensure resulting inventory stays within ±max_inventory implied by notional."""
        # We use max_gross_notional / mid_price as an implied inventory cap.
        max_inv = self.limits.max_gross_notional / max(portfolio.mid_price, 1.0)
        new_inv = portfolio.inventory + order.signed_size
        if abs(new_inv) > max_inv:
            return ValidationResult(
                approved=False,
                reason=RejectionReason.POSITION_LIMIT,
                detail=(
                    f"New inventory {new_inv:.0f} would exceed implied limit "
                    f"±{max_inv:.0f} shares for {order.symbol}."
                ),
            )
        return ValidationResult(approved=True)

    def _check_notional_limit(self, order: Order, portfolio: PortfolioState) -> ValidationResult:
        """Gross notional after fill must be ≤ max_gross_notional."""
        new_inv = portfolio.inventory + order.signed_size
        new_notional = abs(new_inv) * portfolio.mid_price
        if new_notional > self.limits.max_gross_notional:
            return ValidationResult(
                approved=False,
                reason=RejectionReason.NOTIONAL_LIMIT,
                detail=(
                    f"Gross notional ${new_notional:,.0f} would exceed limit "
                    f"${self.limits.max_gross_notional:,.0f}."
                ),
            )
        return ValidationResult(approved=True)

    def _check_leverage_limit(self, order: Order, portfolio: PortfolioState) -> ValidationResult:
        new_inv = portfolio.inventory + order.signed_size
        equity = portfolio.cash + new_inv * portfolio.mid_price
        new_notional = abs(new_inv) * portfolio.mid_price
        leverage = new_notional / max(abs(equity), 1.0)
        if leverage > self.limits.max_leverage:
            return ValidationResult(
                approved=False,
                reason=RejectionReason.LEVERAGE_LIMIT,
                detail=(
                    f"Leverage {leverage:.2f}x would exceed limit "
                    f"{self.limits.max_leverage:.2f}x."
                ),
            )
        return ValidationResult(approved=True)

    def _check_var_limit(self, order: Order, portfolio: PortfolioState) -> ValidationResult:
        """Estimate post-trade VaR and reject if it exceeds the limit."""
        hypo = {"side": order.signed_size / max(order.size, 1), "size": order.size}
        estimated_var = self.risk_engine.estimate_var(portfolio, hypo)
        if estimated_var > self.limits.var_limit_usd:
            return ValidationResult(
                approved=False,
                reason=RejectionReason.VAR_LIMIT,
                detail=(
                    f"Estimated 95% VaR ${estimated_var:,.0f} would exceed limit "
                    f"${self.limits.var_limit_usd:,.0f}."
                ),
            )
        return ValidationResult(approved=True)

    def _check_drawdown_limit(
        self, order: Order, portfolio: PortfolioState
    ) -> ValidationResult:
        """Halt trading if intraday drawdown exceeds the hard limit."""
        dd = portfolio.current_drawdown
        if dd > self.limits.max_intraday_drawdown_usd:
            return ValidationResult(
                approved=False,
                reason=RejectionReason.DRAWDOWN_LIMIT,
                detail=(
                    f"Intraday drawdown ${dd:,.0f} exceeds limit "
                    f"${self.limits.max_intraday_drawdown_usd:,.0f}. Trading halted."
                ),
            )
        return ValidationResult(approved=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_rejection(self, order: Order, result: ValidationResult) -> None:
        reason_key = result.reason.value if result.reason else "UNKNOWN"
        self._rejection_counts[reason_key] = self._rejection_counts.get(reason_key, 0) + 1
        logger.warning(
            "ORDER REJECTED | agent=%s symbol=%s side=%s size=%d | reason=%s | %s",
            order.agent_id,
            order.symbol,
            order.side.name,
            order.size,
            reason_key,
            result.detail,
        )
