"""Perp order-boundary guard — spec docs/plans/07 §12 (Codex live-path reqs #2 + #3).

Defense-in-depth for the LIVE order path. The shadow machine runs the pre-trade
gates (`perp_gates.check_all`) as a pre-flight, but two gaps the Codex review
flagged only bite once an adapter can actually place an order:

  * #2 — nothing STRUCTURALLY stops a future live caller from placing an order
    that skipped `check_all`.
  * #3 — `PreTradeGuard`'s venue check uses the broad GLOBAL allowlist, so a
    miswired perp broker pointed at a globally-legal-but-perp-illegal venue
    (e.g. coinbase-spot, kraken-futures) would pass.

This module is the fail-closed gate the live `submit_order` path MUST call
immediately before placing EITHER leg. Raising here means no order is placed.
Pure + deterministic.
"""
from __future__ import annotations

from .perp_gates import PERP_VENUES, GateResult


class PerpOrderRefused(Exception):
    """Raised at the order boundary when an order must not be placed."""


def assert_perp_venue(venue: str) -> None:
    """Fail-closed venue check: refuse anything not exactly a perp venue.

    Empty/None/whitespace is refused (fail-closed), as is any venue outside
    ``{kalshi-perp, coinbase-futures}`` — including venues that pass the broad
    global allowlist but are not US-legal perp surfaces.
    """
    if not venue or venue.strip().lower() not in PERP_VENUES:
        raise PerpOrderRefused(
            f"venue {venue!r} is not a perp venue {sorted(PERP_VENUES)} — order refused"
        )


def authorize_leg(venue: str, gate_result: GateResult | None) -> None:
    """Authorize one leg for live placement, or raise ``PerpOrderRefused``.

    The live ``submit_order`` path calls this before every order. BOTH must hold:
      * the venue is perp-allowlisted (req #3), and
      * the pre-trade gates ran and passed for this proposal (req #2) — a missing
        or failed ``GateResult`` refuses the order.
    """
    assert_perp_venue(venue)
    if gate_result is None:
        raise PerpOrderRefused(
            "no pre-trade gate result — order refused (gates must run before placement)"
        )
    if not gate_result.passed:
        n = len(gate_result.failures)
        raise PerpOrderRefused(
            f"pre-trade gates did not pass ({n} failure(s)) — order refused"
        )
