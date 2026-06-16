"""Tests for the gated perp executor — spec docs/plans/07 §12 (Codex #2 wired structural).

Proves no order is placed unless the pre-trade gates pass and both legs are
perp-allowlisted, and that a valid proposal executes atomically.
"""
from __future__ import annotations

import dataclasses

import pytest

from quant_engine.execution.guards import PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker
from quant_engine.execution.perp_executor import GatedPerpExecutor
from quant_engine.execution.perp_order_guard import PerpOrderRefused
from quant_engine.execution.perp_strategy import (
    PerpMarketInputs,
    build_orders,
    build_proposal,
)
from quant_engine.execution.two_leg import TwoLegOutcome


def _inputs(**overrides) -> PerpMarketInputs:
    base = dict(
        instrument="BTCUSD-PERP",
        long_venue="kalshi-perp",
        short_venue="coinbase-futures",
        long_funding_annual=0.0,
        short_funding_annual=0.06,
        long_mark_usd=63_000.0,
        short_mark_usd=63_000.0,
        long_liq_buffer_pct=30.0,
        short_liq_buffer_pct=30.0,
    )
    base.update(overrides)
    return PerpMarketInputs(**base)


def _broker() -> PaperBroker:
    guard = PreTradeGuard(
        allowed_instruments={"BTCUSD-PERP"},
        max_notional_usd=3_000.0,
        max_position_qty=10.0,
    )
    return PaperBroker(guard=guard)


def _executor():
    return GatedPerpExecutor(long_broker=_broker(), short_broker=_broker(), mode="shadow")


def test_valid_proposal_executes_atomically():
    inputs = _inputs()
    proposal = build_proposal(inputs)
    long_order, short_order = build_orders(inputs, proposal)

    ex = _executor()
    result = ex.execute(proposal, long_order, short_order)

    assert result.outcome is TwoLegOutcome.BOTH_FILLED
    assert result.is_safe
    assert ex.long_broker.position("BTCUSD-PERP") > 0
    assert ex.short_broker.position("BTCUSD-PERP") < 0


def test_failed_gates_refuse_and_place_nothing():
    inputs = _inputs()
    proposal = build_proposal(inputs)
    long_order, short_order = build_orders(inputs, proposal)
    # Break a gate: short leg liq buffer below the 20% floor → check_all fails.
    bad = dataclasses.replace(proposal, short_liq_buffer_pct=5.0)

    ex = _executor()
    with pytest.raises(PerpOrderRefused):
        ex.execute(bad, long_order, short_order)
    # Nothing placed on refusal.
    assert ex.long_broker.fills == []
    assert ex.short_broker.fills == []


def test_off_allowlist_venue_refused():
    inputs = _inputs()
    proposal = build_proposal(inputs)
    long_order, short_order = build_orders(inputs, proposal)
    bad = dataclasses.replace(proposal, short_venue="coinbase-intx")

    ex = _executor()
    with pytest.raises(PerpOrderRefused):
        ex.execute(bad, long_order, short_order)
    assert ex.long_broker.fills == []


def test_daily_loss_cap_refuses_entry():
    inputs = _inputs()
    proposal = build_proposal(inputs)
    long_order, short_order = build_orders(inputs, proposal)

    ex = _executor()
    with pytest.raises(PerpOrderRefused):
        ex.execute(proposal, long_order, short_order, day_pnl_usd=-250.0)  # past -$200 cap
    assert ex.long_broker.fills == []
