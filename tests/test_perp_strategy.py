"""Unit tests for the perp carry strategy core — spec docs/plans/07 §3.

Proves the decision layer emits gate-valid, delta-neutral proposals; respects the
pre-registered no-trade threshold; and that its orders execute cleanly through the
atomic two-leg executor in shadow mode (PaperBroker both legs — places nothing).
"""
from __future__ import annotations

import dataclasses

import pytest

from quant_engine.execution.guards import PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker
from quant_engine.execution.perp_gates import PerpGateConfig, check_all
from quant_engine.execution.perp_strategy import (
    PerpMarketInputs,
    PerpStrategyConfig,
    build_orders,
    build_proposal,
    captured_carry_annual,
)
from quant_engine.execution.two_leg import TwoLegExecutor, TwoLegOutcome


def _inputs(**overrides) -> PerpMarketInputs:
    base = dict(
        instrument="BTCUSD-PERP",
        long_venue="kalshi-perp",
        short_venue="coinbase-futures",
        long_funding_annual=0.0,        # Kalshi pinned in the dead band
        short_funding_annual=0.06,      # Coinbase CFM pays ~+6%/yr
        long_mark_usd=63_000.0,
        short_mark_usd=63_000.0,
        long_liq_buffer_pct=30.0,
        short_liq_buffer_pct=30.0,
    )
    base.update(overrides)
    return PerpMarketInputs(**base)


# ---------------------------------------------------------------------------
# Carry math + the no-trade threshold
# ---------------------------------------------------------------------------

def test_captured_carry_is_short_minus_long():
    assert captured_carry_annual(_inputs()) == pytest.approx(0.06)


def test_thin_edge_returns_no_trade():
    assert build_proposal(_inputs(short_funding_annual=0.02)) is None


def test_kalshi_out_of_dead_band_kills_edge():
    # If the long leg is no longer pinned at 0, the differential collapses.
    assert build_proposal(_inputs(long_funding_annual=0.05)) is None


def test_non_positive_mark_raises():
    with pytest.raises(ValueError, match="non-positive mark"):
        build_proposal(_inputs(short_mark_usd=0.0))


# ---------------------------------------------------------------------------
# Proposal shape: delta-neutral, within caps, gate-valid
# ---------------------------------------------------------------------------

def test_proposal_is_delta_neutral():
    p = build_proposal(_inputs())
    assert p is not None
    assert p.long_qty_btc == pytest.approx(p.short_qty_btc)


def test_proposal_passes_all_gates():
    """The strategy's output must satisfy the very gates that guard the executor."""
    p = build_proposal(_inputs())
    result = check_all(p, PerpGateConfig())
    assert result.passed, result.failures


def test_notional_never_exceeds_target_when_marks_differ():
    # Size off the larger mark → neither leg can exceed the target notional.
    cfg = PerpStrategyConfig(target_leg_notional_usd=2_500.0)
    p = build_proposal(_inputs(long_mark_usd=63_000.0, short_mark_usd=64_000.0), cfg)
    assert p is not None
    assert p.long_notional_usd <= 2_500.0 + 1e-6
    assert p.short_notional_usd <= 2_500.0 + 1e-6


def test_funding_diff_recorded_on_proposal():
    p = build_proposal(_inputs())
    assert p.funding_diff_annual == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def test_build_orders_long_buy_short_sell():
    inputs = _inputs()
    p = build_proposal(inputs)
    long_order, short_order = build_orders(inputs, p)

    assert long_order.side == "buy"
    assert long_order.instrument == "BTCUSD-PERP"
    assert long_order.price == pytest.approx(inputs.long_mark_usd)
    assert short_order.side == "sell"
    assert short_order.price == pytest.approx(inputs.short_mark_usd)
    assert long_order.qty == pytest.approx(short_order.qty)


# ---------------------------------------------------------------------------
# Capstone: strategy → orders → shadow two-leg execution (zero capital)
# ---------------------------------------------------------------------------

def _shadow_broker() -> PaperBroker:
    guard = PreTradeGuard(
        allowed_instruments={"BTCUSD-PERP"},
        max_notional_usd=3_000.0,
        max_position_qty=1.0,
    )
    return PaperBroker(guard=guard)


def test_strategy_executes_delta_neutral_in_shadow():
    inputs = _inputs()
    proposal = build_proposal(inputs)
    long_order, short_order = build_orders(inputs, proposal)

    long_broker = _shadow_broker()
    short_broker = _shadow_broker()
    executor = TwoLegExecutor(long_broker=long_broker, short_broker=short_broker, mode="shadow")

    result = executor.execute(long_order, short_order)

    assert result.outcome is TwoLegOutcome.BOTH_FILLED
    assert result.is_safe
    # Opposite positions of equal size → delta-neutral across the two legs.
    long_pos = long_broker.position("BTCUSD-PERP")
    short_pos = short_broker.position("BTCUSD-PERP")
    assert long_pos > 0 and short_pos < 0
    assert long_pos + short_pos == pytest.approx(0.0)
