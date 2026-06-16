"""Tests for the perp order-boundary guard — spec docs/plans/07 §12 (Codex #2 + #3)."""
from __future__ import annotations

import pytest

from quant_engine.execution.perp_gates import GateResult
from quant_engine.execution.perp_order_guard import (
    PerpOrderRefused,
    assert_perp_venue,
    authorize_leg,
)


# ---------------------------------------------------------------------------
# Req #3 — perp-venue fail-closed
# ---------------------------------------------------------------------------

def test_perp_venues_allowed():
    assert_perp_venue("kalshi-perp")
    assert_perp_venue("coinbase-futures")
    assert_perp_venue("Coinbase-Futures")   # case-insensitive


def test_globally_legal_but_non_perp_venue_refused():
    # These pass the broad global allowlist but are NOT perp surfaces.
    for venue in ("coinbase-spot", "kraken-futures", "coinbase-intx", "polymarket"):
        with pytest.raises(PerpOrderRefused, match="not a perp venue"):
            assert_perp_venue(venue)


def test_empty_or_none_venue_refused_fail_closed():
    for venue in ("", "   ", None):
        with pytest.raises(PerpOrderRefused):
            assert_perp_venue(venue)


# ---------------------------------------------------------------------------
# Req #2 — gates must have run and passed
# ---------------------------------------------------------------------------

def test_authorize_passes_with_allowlisted_venue_and_passed_gates():
    authorize_leg("kalshi-perp", GateResult(passed=True, failures=[]))


def test_authorize_refused_when_gates_failed():
    failed = GateResult(passed=False, failures=["net delta 0.01 BTC exceeds eps"])
    with pytest.raises(PerpOrderRefused, match="did not pass"):
        authorize_leg("coinbase-futures", failed)


def test_authorize_refused_when_no_gate_result():
    with pytest.raises(PerpOrderRefused, match="no pre-trade gate result"):
        authorize_leg("kalshi-perp", None)


def test_authorize_refused_on_bad_venue_even_with_passed_gates():
    # Venue check is first and fail-closed — passing gates can't rescue a bad venue.
    with pytest.raises(PerpOrderRefused, match="not a perp venue"):
        authorize_leg("coinbase-spot", GateResult(passed=True, failures=[]))


def test_authorize_refused_on_inconsistent_gate_result():
    # passed=True but failures present is logically inconsistent — refuse the order.
    with pytest.raises(PerpOrderRefused):
        authorize_leg("kalshi-perp", GateResult(passed=True, failures=["sneaky"]))
