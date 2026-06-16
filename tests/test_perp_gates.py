"""Perp pre-trade gates — spec docs/plans/07 §4.2."""
from __future__ import annotations

import dataclasses

from quant_engine.execution.perp_gates import (
    PerpGateConfig,
    PerpTradeProposal,
    check_all,
    gate_allowlist,
    gate_daily_loss,
    gate_edge_clears_fees,
    gate_leverage,
    gate_liquidation_buffer,
    gate_max_position,
    gate_net_delta,
)


def _valid() -> PerpTradeProposal:
    # Delta-neutral $2.5K/leg, 2.5x, 30% buffer; +6%/yr carry held a quarter clears
    # ~$20 round-trip fees: 2500 * 0.06 * (2190/8760) = $37.5 >= $20 + $5 buffer.
    return PerpTradeProposal(
        long_venue="kalshi-perp",
        long_qty_btc=0.04,
        long_notional_usd=2_500.0,
        long_leverage=2.5,
        long_liq_buffer_pct=30.0,
        short_venue="coinbase-futures",
        short_qty_btc=0.04,
        short_notional_usd=2_500.0,
        short_leverage=2.5,
        short_liq_buffer_pct=30.0,
        long_funding_annual=0.0,
        short_funding_annual=0.06,
        funding_diff_annual=0.06,
        hold_hours=2_190.0,
        round_trip_fees_usd=20.0,
    )


CFG = PerpGateConfig()


def test_valid_proposal_passes_all_gates():
    res = check_all(_valid(), CFG, day_pnl_usd=0.0)
    assert res.passed
    assert res.failures == []


def test_allowlist_rejects_intx():
    p = dataclasses.replace(_valid(), short_venue="coinbase-intx")
    assert gate_allowlist(p, CFG)[0] is False
    assert not check_all(p, CFG).passed


def test_net_delta_gate():
    p = dataclasses.replace(_valid(), short_qty_btc=0.05)  # net -0.01 BTC
    assert gate_net_delta(p, CFG)[0] is False


def test_leverage_gate():
    p = dataclasses.replace(_valid(), long_leverage=5.0)  # > 3x cap
    assert gate_leverage(p, CFG)[0] is False


def test_edge_clears_fees_gate():
    p = dataclasses.replace(_valid(), hold_hours=24.0)  # 1 day → funding ~$0.41 < fees
    assert gate_edge_clears_fees(p, CFG)[0] is False


def test_edge_gate_passes_with_long_enough_hold():
    assert gate_edge_clears_fees(_valid(), CFG)[0] is True


def test_edge_gate_uses_per_leg_notionals_not_long_proxy():
    # SHORT $2.5K collecting +6%, LONG only $1K paying 0%. The funding we actually
    # collect is on the SHORT notional. The old long-notional proxy
    # (1000 * 0.06 * 0.25 = $15) would wrongly REJECT; per-leg
    # (2500 * 0.06 * 0.25 = $37.5) correctly ACCEPTS.
    p = dataclasses.replace(
        _valid(),
        long_notional_usd=1_000.0,
        short_notional_usd=2_500.0,
        long_funding_annual=0.0,
        short_funding_annual=0.06,
    )
    assert gate_edge_clears_fees(p, CFG)[0] is True


def test_liquidation_buffer_gate():
    p = dataclasses.replace(_valid(), long_liq_buffer_pct=10.0)  # < 20%
    assert gate_liquidation_buffer(p, CFG)[0] is False


def test_daily_loss_kill_switch():
    assert gate_daily_loss(_valid(), CFG, day_pnl_usd=-250.0)[0] is False
    assert gate_daily_loss(_valid(), CFG, day_pnl_usd=-50.0)[0] is True
    assert not check_all(_valid(), CFG, day_pnl_usd=-250.0).passed


def test_max_position_gate():
    p = dataclasses.replace(_valid(), short_notional_usd=5_000.0)  # > $2.5K cap
    assert gate_max_position(p, CFG)[0] is False


def test_check_all_aggregates_multiple_failures():
    p = dataclasses.replace(_valid(), short_venue="coinbase-intx", long_leverage=9.0)
    res = check_all(p, CFG)
    assert res.passed is False
    assert len(res.failures) >= 2


def test_non_finite_field_fails_closed():
    # NaN compares False against every threshold; without the finite gate this passes.
    p = dataclasses.replace(_valid(), long_qty_btc=float("nan"))
    res = check_all(p, CFG)
    assert res.passed is False
    assert any("non-finite" in f for f in res.failures)


def test_non_finite_day_pnl_fails_closed():
    res = check_all(_valid(), CFG, day_pnl_usd=float("inf"))
    assert res.passed is False
    assert any("day_pnl_usd" in f for f in res.failures)


def test_non_string_venue_fails_closed():
    # A None/non-string venue must REJECT, not raise — keeps check_all fail-closed.
    p = dataclasses.replace(_valid(), long_venue=None)
    res = check_all(p, CFG)
    assert res.passed is False
    assert any("allowlist" in f for f in res.failures)
