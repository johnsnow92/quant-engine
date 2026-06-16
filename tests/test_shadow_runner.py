"""End-to-end shadow-runner tests — spec docs/plans/07 §4.6 / build step 7.

The whole machine wired together with zero capital: read → propose → gate →
shadow-execute → reconcile. Asserts every decision path and that the live
read-only adapters are never asked to place an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from quant_engine.execution.perp_gates import PerpGateConfig
from quant_engine.execution.shadow_runner import (
    BLOCKED_BY_GATES,
    EXECUTED_SHADOW,
    NO_TRADE_THIN_EDGE,
    PerpShadowRunner,
)
from quant_engine.execution.two_leg import TwoLegOutcome


@dataclass
class _FakeReader:
    """Read-only stand-in for the live adapters, plus a tripwire submit_order."""
    funding: float
    mark: float
    buffer: float

    def funding_rate_annual(self, instrument: str) -> float:
        return self.funding

    def mark_price(self, instrument: str) -> float:
        return self.mark

    def margin_buffer_pct(self) -> float:
        return self.buffer


def _runner(long_funding=0.0, short_funding=0.06, long_buf=30.0, short_buf=30.0, **kw):
    long_reader = _FakeReader(funding=long_funding, mark=63_000.0, buffer=long_buf)
    short_reader = _FakeReader(funding=short_funding, mark=63_000.0, buffer=short_buf)
    # Tripwires: if the runner ever tries to place on a live adapter, fail loudly.
    long_reader.submit_order = MagicMock(side_effect=AssertionError("placed on live reader"))
    short_reader.submit_order = MagicMock(side_effect=AssertionError("placed on live reader"))
    return PerpShadowRunner(long_reader=long_reader, short_reader=short_reader, **kw)


# ---------------------------------------------------------------------------
# Healthy carry → executes in shadow, reconciles clean, places nothing real
# ---------------------------------------------------------------------------

def test_healthy_carry_executes_in_shadow():
    runner = _runner()
    decision = runner.run_cycle()

    assert decision.action == EXECUTED_SHADOW
    assert decision.two_leg_result.outcome is TwoLegOutcome.BOTH_FILLED
    assert decision.two_leg_result.is_safe
    assert decision.recon_result.ok
    assert decision.recon_result.should_flatten is False
    # No order ever routed to a live adapter.
    runner.long_reader.submit_order.assert_not_called()
    runner.short_reader.submit_order.assert_not_called()


def test_executed_position_is_delta_neutral():
    decision = _runner().run_cycle()
    snap_delta = (
        decision.proposal.long_qty_btc - decision.proposal.short_qty_btc
    )
    assert snap_delta == pytest.approx(0.0)
    assert decision.recon_result.breaches == []


# ---------------------------------------------------------------------------
# Thin edge → no trade
# ---------------------------------------------------------------------------

def test_thin_edge_no_trade():
    runner = _runner(short_funding=0.01)   # carry 1%/yr < 3% threshold
    decision = runner.run_cycle()

    assert decision.action == NO_TRADE_THIN_EDGE
    assert decision.two_leg_result is None
    runner.long_reader.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# Gates block a thin-margin or loss-capped entry → no execution
# ---------------------------------------------------------------------------

def test_low_liq_buffer_blocked_by_gates():
    runner = _runner(short_buf=5.0)   # below the 20% liq-buffer floor
    decision = runner.run_cycle()

    assert decision.action == BLOCKED_BY_GATES
    assert decision.two_leg_result is None
    assert any("liq buffer" in f for f in decision.gate_result.failures)


def test_daily_loss_cap_blocks_entry():
    runner = _runner()
    decision = runner.run_cycle(day_pnl_usd=-250.0)   # past the -$200 kill cap

    assert decision.action == BLOCKED_BY_GATES
    assert any("daily P&L" in f for f in decision.gate_result.failures)
    assert decision.two_leg_result is None


# ---------------------------------------------------------------------------
# Summaries are readable for each path
# ---------------------------------------------------------------------------

def test_summaries_are_human_readable():
    assert "executed" in _runner().run_cycle().summary()
    assert "no trade" in _runner(short_funding=0.01).run_cycle().summary()
    assert "BLOCKED" in _runner(short_buf=5.0).run_cycle().summary()


def test_custom_gate_config_is_honored():
    # Raise the liq-buffer floor above the supplied buffer → blocked.
    runner = _runner(short_buf=25.0, gate_cfg=PerpGateConfig(min_liq_buffer_pct=40.0))
    assert runner.run_cycle().action == BLOCKED_BY_GATES
