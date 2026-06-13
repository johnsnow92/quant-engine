"""Venue-legality allowlist — the hard compliance gate.

Operating rule 2 (portfolio CLAUDE.md): only US-legal, CFTC-regulated venues may
receive orders from this Michigan-based operator. This module is **default-deny**:
a venue that is not explicitly allowlisted is rejected — no exceptions, no LLM in
the path. North-star guardrail KPI: ``off-allowlist orders = 0``.

This is a *venue*-level gate. Market-category restrictions (e.g. Kalshi non-sports
only) are a separate, finer gate and are NOT enforced here.

Allowlisted (execution permitted):
    kalshi            CFTC-regulated event/derivatives exchange
    ibkr-forecastex   CFTC-regulated event contracts via Interactive Brokers
    gemini            CFTC-regulated event contracts (Gemini)
    coinbase-futures  Coinbase Financial Markets — CFTC-regulated nano/standard futures
    coinbase-spot     Coinbase Advanced Trade spot (US-regulated)
    kraken-futures    Kraken's CFTC-regulated US futures/perps
    cme               CME-listed futures (reference/hedge legs)

Explicitly BLOCKED (known-tempting — must never receive an order):
    coinbase-intx     Coinbase International perpetuals — not US-retail legal
    agentic-wallet    on-chain agentic wallet routing — off-allowlist
    polymarket        execution disabled from Michigan (read-only data only)
    betfair / smarkets / sxbet / matchbook   US-restricted exchanges/books
"""
from __future__ import annotations

from .guards import GuardRejection

VENUE_KALSHI = "kalshi"
VENUE_IBKR_FORECASTEX = "ibkr-forecastex"
VENUE_GEMINI = "gemini"
VENUE_COINBASE_FUTURES = "coinbase-futures"
VENUE_COINBASE_SPOT = "coinbase-spot"
VENUE_KRAKEN_FUTURES = "kraken-futures"
VENUE_CME = "cme"

VENUE_COINBASE_INTX = "coinbase-intx"

# US-legal, CFTC-regulated venues that may receive live orders.
ALLOWLISTED_VENUES: frozenset[str] = frozenset(
    {
        VENUE_KALSHI,
        VENUE_IBKR_FORECASTEX,
        VENUE_GEMINI,
        VENUE_COINBASE_FUTURES,
        VENUE_COINBASE_SPOT,
        VENUE_KRAKEN_FUTURES,
        VENUE_CME,
    }
)

# Explicit deny list with the reason, for clearer rejections than bare default-deny.
BLOCKED_VENUES: dict[str, str] = {
    VENUE_COINBASE_INTX: "Coinbase International perpetuals are not US-retail legal "
    "(Michigan); use the CFTC-regulated coinbase-futures venue instead",
    "agentic-wallet": "on-chain agentic wallet routing is off-allowlist",
    "polymarket": "Polymarket execution is disabled from Michigan (read-only data only)",
    "betfair": "Betfair is US-restricted",
    "smarkets": "Smarkets is US-restricted",
    "sxbet": "SX Bet is US-restricted",
    "matchbook": "Matchbook is US-restricted",
}


class VenueNotAllowed(GuardRejection):
    """Raised when an order would route to an off-allowlist venue.

    Subclasses GuardRejection so existing ``except GuardRejection`` paths and the
    kill-switch boundary treat it as a hard pre-trade rejection.
    """


def normalize_venue(venue: str) -> str:
    """Canonicalize a venue id: trimmed, lowercased, spaces/underscores → hyphen."""
    return venue.strip().lower().replace("_", "-").replace(" ", "-")


def is_venue_allowed(venue: str) -> bool:
    """True iff ``venue`` is on the execution allowlist (default-deny)."""
    return normalize_venue(venue) in ALLOWLISTED_VENUES


def assert_venue_allowed(venue: str) -> str:
    """Return the normalized venue if allowlisted, else raise ``VenueNotAllowed``.

    Default-deny: any venue not on ``ALLOWLISTED_VENUES`` is rejected. If the venue
    is on the explicit ``BLOCKED_VENUES`` deny list, the reason is included.
    """
    canonical = normalize_venue(venue)
    if canonical in ALLOWLISTED_VENUES:
        return canonical
    reason = BLOCKED_VENUES.get(canonical, "venue is not on the US-legal allowlist")
    raise VenueNotAllowed(f"venue {canonical!r} blocked: {reason}")
