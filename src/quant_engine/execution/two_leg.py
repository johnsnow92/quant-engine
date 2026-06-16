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


_FLAT_EPS = 1e-9


def _reverse(order: Order) -> Order:
    """The reduce-only MARKET order that flattens ``order`` — Codex live-path req #1.

    A live unwind must be reduce-only at market (slippage-capped by the venue), NOT a
    limit at the original entry price that a moving market can leave unfilled, which
    would strand the leg naked. ``price`` is kept only as the entry reference for the
    Fill; the live broker ignores it for a market order.
    """
    return Order(
        instrument=order.instrument,
        side="sell" if order.side == "buy" else "buy",
        qty=order.qty,
        price=order.price,
        reduce_only=True,
        order_type="market",
    )


def _leg_position(broker: object, instrument: str) -> float | None:
    """The broker's signed position in ``instrument``.

    Returns None ONLY when the broker exposes no ``position()`` method — a static
    capability gap the caller treats as assume-flat. If ``position()`` EXISTS but
    raises, the exception propagates: a transient query failure during verify-flat
    must not be read as 'flat', so the caller escalates to NAKED_LEG.
    """
    pos_fn = getattr(broker, "position", None)
    if pos_fn is None:
        return None
    return pos_fn(instrument)


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
        except Exception as exc:
            log.critical(
                "[%s] NAKED LEG — short failed AND unwind submit failed (short=%s, unwind=%s)",
                self.mode,
                short_error,
                exc,
            )
            return TwoLegResult(
                TwoLegOutcome.NAKED_LEG,
                long_fill=long_fill,
                error=f"short_failed={short_error}; unwind_failed={exc}",
            )

        # Verify-flat: a 'successful' unwind submit is not proof the leg closed. A
        # partial or no-op reduce that leaves exposure is still a naked leg — confirm
        # the broker reports the long flat before calling it UNWOUND. A position
        # query that RAISES means flatness is unverifiable → escalate to NAKED_LEG
        # rather than assume flat (only a missing position() method assumes flat).
        try:
            residual = _leg_position(self.long_broker, long_order.instrument)
        except Exception as exc:
            log.critical(
                "[%s] NAKED LEG — unwind submitted but flatness UNVERIFIABLE "
                "(position query failed: %s)",
                self.mode,
                exc,
            )
            return TwoLegResult(
                TwoLegOutcome.NAKED_LEG,
                long_fill=long_fill,
                unwind_fill=unwind,
                error=f"unwind flatness unverifiable: position query failed ({exc}); "
                      f"short_failed={short_error}",
            )
        if residual is not None and abs(residual) > _FLAT_EPS:
            log.critical(
                "[%s] NAKED LEG — unwind did not flatten long leg: residual=%.8f",
                self.mode,
                residual,
            )
            return TwoLegResult(
                TwoLegOutcome.NAKED_LEG,
                long_fill=long_fill,
                unwind_fill=unwind,
                error=f"unwind did not flatten long leg: residual={residual} (short_failed={short_error})",
            )

        log.info("[%s] long leg unwound — verified flat", self.mode)
        return TwoLegResult(
            TwoLegOutcome.UNWOUND,
            long_fill=long_fill,
            unwind_fill=unwind,
            error=short_error,
        )
