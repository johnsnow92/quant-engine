"""Atomic two-leg perp executor — spec docs/plans/07-perp-executor.md §4.3.

Places a long leg and a short leg. If one fills and the other fails, the filled
leg is IMMEDIATELY unwound so a naked (one-legged) position is never held past the
attempt — the hard guardrail for the dead-band carry trade (naked-leg events = 0).
If even the unwind fails, the result is NAKED_LEG (CRITICAL) rather than a silent
exposure, so the caller / reconciliation daemon can act.

Venue-agnostic and deterministic (no LLM): pass PaperBroker for **shadow** mode
(places nothing real), the live Kalshi-perp / Coinbase-CFM brokers for live. Each
broker enforces its own pre-trade guard; cross-leg gates (net-delta, edge-clears-
fees) are a separate pre-flight (spec §4.2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from .guards import Order
from .paper_broker import Fill

log = logging.getLogger(__name__)


class TwoLegOutcome(str, Enum):
    BOTH_FILLED = "both_filled"   # delta-neutral position opened
    LONG_FAILED = "long_failed"   # long never filled; nothing placed, nothing naked
    UNWOUND = "unwound"           # short failed; long filled then unwound cleanly
    NAKED_LEG = "naked_leg"       # short failed AND unwind failed → CRITICAL exposure


@dataclass
class TwoLegResult:
    outcome: TwoLegOutcome
    long_fill: Fill | None = None
    short_fill: Fill | None = None
    unwind_fill: Fill | None = None
    error: str | None = None

    @property
    def is_safe(self) -> bool:
        """True unless we are holding a naked (one-legged) position."""
        return self.outcome is not TwoLegOutcome.NAKED_LEG


def _reverse(order: Order) -> Order:
    """The order that flattens ``order`` (same instrument, qty, price; opposite side)."""
    return Order(
        instrument=order.instrument,
        side="sell" if order.side == "buy" else "buy",
        qty=order.qty,
        price=order.price,
    )


@dataclass
class TwoLegExecutor:
    """Coordinates the long and short legs of the delta-neutral carry trade.

    ``long_broker`` / ``short_broker`` each implement ``submit_order(Order) -> Fill``
    and raise on rejection (PaperBroker, CoinbaseBroker, the Kalshi-perp adapter).
    ``mode`` is informational for logging — the brokers enforce reality (shadow =
    PaperBroker on both legs).
    """

    long_broker: object
    short_broker: object
    mode: str = "shadow"

    def execute(self, long_order: Order, short_order: Order) -> TwoLegResult:
        # Leg 1 — long. If it never fills, nothing is placed and nothing is naked.
        try:
            long_fill = self.long_broker.submit_order(long_order)
        except Exception as exc:
            log.warning("[%s] long leg failed, nothing placed: %s", self.mode, exc)
            return TwoLegResult(TwoLegOutcome.LONG_FAILED, error=str(exc))

        # Leg 2 — short. If it fails, the long is naked → unwind it immediately.
        try:
            short_fill = self.short_broker.submit_order(short_order)
        except Exception as exc:
            log.error(
                "[%s] short leg failed after long filled — unwinding long: %s",
                self.mode,
                exc,
            )
            return self._unwind_long(long_order, long_fill, str(exc))

        log.info(
            "[%s] both legs filled — long %s @ %.4f / short %s @ %.4f",
            self.mode,
            long_order.instrument,
            long_fill.price,
            short_order.instrument,
            short_fill.price,
        )
        return TwoLegResult(
            TwoLegOutcome.BOTH_FILLED, long_fill=long_fill, short_fill=short_fill
        )

    def _unwind_long(
        self, long_order: Order, long_fill: Fill, short_error: str
    ) -> TwoLegResult:
        try:
            unwind = self.long_broker.submit_order(_reverse(long_order))
            log.info("[%s] long leg unwound cleanly — flat", self.mode)
            return TwoLegResult(
                TwoLegOutcome.UNWOUND,
                long_fill=long_fill,
                unwind_fill=unwind,
                error=short_error,
            )
        except Exception as exc:
            log.critical(
                "[%s] NAKED LEG — short failed AND unwind failed (short=%s, unwind=%s)",
                self.mode,
                short_error,
                exc,
            )
            return TwoLegResult(
                TwoLegOutcome.NAKED_LEG,
                long_fill=long_fill,
                error=f"short_failed={short_error}; unwind_failed={exc}",
            )
