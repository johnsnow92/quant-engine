"""Pre-trade risk guards.

Every order must pass the guard layer before any broker (paper or live) acts on
it. Guards are the kill-switch boundary: position limits, notional caps, an
instrument allowlist, an optional venue-legality allowlist, and a global halt.
A rejected order raises and never fills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class GuardRejection(Exception):
    """Raised when an order violates a pre-trade risk limit."""


@dataclass
class Order:
    instrument: str
    side: str  # 'buy' or 'sell'
    qty: float
    price: float
    reduce_only: bool = False   # live: only reduce an existing position, never flip it
    order_type: str = "limit"   # "limit" (default) | "market" — live unwind uses "market"

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if not math.isfinite(self.qty):
            raise ValueError(f"qty must be finite, got {self.qty!r}")
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if not math.isfinite(self.price):
            raise ValueError(f"price must be finite, got {self.price!r}")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.order_type not in ("limit", "market"):
            raise ValueError(f"order_type must be 'limit' or 'market', got {self.order_type!r}")

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == "buy" else -self.qty


@dataclass
class PreTradeGuard:
    allowed_instruments: set[str]
    max_notional_usd: float
    max_position_qty: float
    kill_switch: bool = False
    venue: str | None = None

    def check(self, order: Order, current_position: float = 0.0) -> Order:
        """Return the order if it passes every limit, else raise GuardRejection."""
        if self.kill_switch:
            raise GuardRejection("kill switch engaged: all trading halted")
        # Venue-legality is the outermost compliance gate (after the kill switch):
        # an order to an off-allowlist venue is rejected before any other check.
        # Imported lazily to avoid a circular import (venue_legality imports
        # GuardRejection from this module).
        if self.venue is not None:
            from .venue_legality import assert_venue_allowed

            assert_venue_allowed(self.venue)
        if order.instrument not in self.allowed_instruments:
            raise GuardRejection(f"instrument {order.instrument} not in allowlist")
        if order.notional > self.max_notional_usd:
            raise GuardRejection(
                f"notional {order.notional:.2f} exceeds max {self.max_notional_usd:.2f}"
            )
        projected = current_position + order.signed_qty
        if abs(projected) > self.max_position_qty:
            raise GuardRejection(
                f"projected position {projected} exceeds max {self.max_position_qty}"
            )
        return order
