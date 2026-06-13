"""Unit tests for the venue-legality allowlist gate.

Guardrail under test: off-allowlist orders = 0. The gate is default-deny and is
wired into PreTradeGuard as the outermost compliance check (after the kill switch).
"""
from __future__ import annotations

import pytest

from quant_engine.execution.guards import GuardRejection, Order, PreTradeGuard
from quant_engine.execution.venue_legality import (
    ALLOWLISTED_VENUES,
    BLOCKED_VENUES,
    VENUE_COINBASE_FUTURES,
    VENUE_COINBASE_INTX,
    VenueNotAllowed,
    assert_venue_allowed,
    is_venue_allowed,
    normalize_venue,
)


# ---------------------------------------------------------------------------
# normalize_venue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Kalshi", "kalshi"),
        ("  COINBASE_FUTURES  ", "coinbase-futures"),
        ("coinbase intx", "coinbase-intx"),
        ("IBKR-ForecastEx", "ibkr-forecastex"),
    ],
)
def test_normalize_venue(raw, expected):
    assert normalize_venue(raw) == expected


# ---------------------------------------------------------------------------
# is_venue_allowed / assert_venue_allowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("venue", sorted(ALLOWLISTED_VENUES))
def test_allowlisted_venues_pass(venue):
    assert is_venue_allowed(venue) is True
    assert assert_venue_allowed(venue) == venue


@pytest.mark.parametrize("venue", sorted(BLOCKED_VENUES))
def test_blocked_venues_rejected_with_reason(venue):
    assert is_venue_allowed(venue) is False
    with pytest.raises(VenueNotAllowed) as exc:
        assert_venue_allowed(venue)
    # The configured reason is surfaced in the rejection message.
    assert BLOCKED_VENUES[venue].split()[0].lower() in str(exc.value).lower()


def test_unknown_venue_default_denied():
    """A venue on neither list is denied (default-deny), not allowed."""
    assert is_venue_allowed("ftx") is False
    with pytest.raises(VenueNotAllowed, match="not on the US-legal allowlist"):
        assert_venue_allowed("ftx")


def test_intx_is_blocked_and_points_to_regulated_alternative():
    with pytest.raises(VenueNotAllowed, match="coinbase-futures"):
        assert_venue_allowed(VENUE_COINBASE_INTX)


def test_venue_not_allowed_is_a_guard_rejection():
    """So existing `except GuardRejection` paths and the kill-switch boundary catch it."""
    assert issubclass(VenueNotAllowed, GuardRejection)


def test_allowlist_and_blocklist_are_disjoint():
    assert ALLOWLISTED_VENUES.isdisjoint(BLOCKED_VENUES.keys())


# ---------------------------------------------------------------------------
# PreTradeGuard integration
# ---------------------------------------------------------------------------

def _guard(**overrides) -> PreTradeGuard:
    base = dict(
        allowed_instruments={"BTCUSD-PERP"},
        max_notional_usd=5_000.0,
        max_position_qty=1.0,
    )
    base.update(overrides)
    return PreTradeGuard(**base)


def test_guard_without_venue_skips_venue_check():
    """Backward compatible: venue=None (default) does no venue check."""
    guard = _guard()  # venue defaults to None
    order = Order("BTCUSD-PERP", "buy", 0.01, 63_000.0)
    assert guard.check(order) is order


def test_guard_with_allowlisted_venue_passes():
    guard = _guard(venue=VENUE_COINBASE_FUTURES)
    order = Order("BTCUSD-PERP", "buy", 0.01, 63_000.0)
    assert guard.check(order) is order


def test_guard_with_offallowlist_venue_rejects_valid_order():
    """The guardrail proof: a fully valid order (instrument + notional + position
    all within limits) is STILL rejected when the guard's venue is off-allowlist."""
    guard = _guard(venue=VENUE_COINBASE_INTX)
    order = Order("BTCUSD-PERP", "buy", 0.01, 63_000.0)  # 630 notional < 5000 cap, OK
    with pytest.raises(VenueNotAllowed):
        guard.check(order)


def test_venue_gate_precedes_instrument_and_notional_checks():
    """Venue legality is the outermost gate: an off-allowlist venue is rejected
    even when the instrument is unknown and the notional is over the cap."""
    guard = _guard(venue="polymarket", max_notional_usd=1.0)
    order = Order("DOGEUSD-PERP", "buy", 1.0, 1_000.0)  # unknown instrument + over cap
    with pytest.raises(VenueNotAllowed):
        guard.check(order)


def test_kill_switch_still_precedes_venue_gate():
    """Kill switch remains the very first check, even before venue legality."""
    guard = _guard(venue=VENUE_COINBASE_INTX, kill_switch=True)
    order = Order("BTCUSD-PERP", "buy", 0.01, 63_000.0)
    with pytest.raises(GuardRejection, match="kill switch"):
        guard.check(order)
