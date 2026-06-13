"""Funding-convention verification — spec docs/plans/07 §4.5."""
from __future__ import annotations

import pytest

from quant_engine.execution.funding_check import (
    FundingObservation,
    expected_funding_usd,
    verify_funding,
)


def _short(realized: float, rate: float = 0.06, elapsed: float = 168.0) -> FundingObservation:
    # Short 0.04 BTC @ $63,000, +6%/yr funding, one week.
    return FundingObservation(
        venue="coinbase-futures",
        position_btc=-0.04,
        mark_price_usd=63_000.0,
        stated_rate_annual=rate,
        elapsed_hours=elapsed,
        realized_funding_usd=realized,
    )


def test_short_receives_long_pays():
    short = _short(realized=0.0)
    assert expected_funding_usd(short) > 0.0          # short receives when rate>0
    long = FundingObservation("kalshi-perp", 0.04, 63_000.0, 0.06, 168.0, 0.0)
    assert expected_funding_usd(long) < 0.0            # long pays when rate>0


def test_negative_rate_flips_direction():
    # Rate negative → longs receive, shorts pay.
    short = _short(realized=0.0, rate=-0.06)
    assert expected_funding_usd(short) < 0.0


def test_correct_funding_passes():
    exp = expected_funding_usd(_short(realized=0.0))
    ok, reason = verify_funding(_short(realized=exp))
    assert ok
    assert reason == ""


def test_sign_mismatch_fails():
    exp = expected_funding_usd(_short(realized=0.0))      # positive (short receives)
    ok, reason = verify_funding(_short(realized=-exp))    # realized has wrong sign
    assert ok is False
    assert "SIGN mismatch" in reason


def test_interval_magnitude_mismatch_fails():
    exp = expected_funding_usd(_short(realized=0.0))
    # 8h-vs-hourly bug makes realized ~3x expected → magnitude fail.
    ok, reason = verify_funding(_short(realized=exp * 3.0))
    assert ok is False
    assert "MAGNITUDE mismatch" in reason


def test_small_diff_within_absolute_tolerance_passes():
    exp = expected_funding_usd(_short(realized=0.0))      # ~ +$2.90
    ok, _ = verify_funding(_short(realized=exp + 0.5))    # $0.50 < $1 abs tolerance
    assert ok


def test_relative_tolerance_passes_at_scale():
    obs = FundingObservation("coinbase-futures", -4.0, 63_000.0, 0.06, 168.0, 0.0)
    exp = expected_funding_usd(obs)                       # large notional → ~$290
    realized = exp * 1.05                                 # 5% off, under 10% rel band
    ok, _ = verify_funding(
        FundingObservation("coinbase-futures", -4.0, 63_000.0, 0.06, 168.0, realized)
    )
    assert ok
    assert exp == pytest.approx(290.0, abs=2.0)
