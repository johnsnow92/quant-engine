"""Tests for the order planner + dry-run path (execution/planner.py)."""
import pytest

from quant_engine.execution.guards import GuardRejection, PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker
from quant_engine.execution.planner import (
    PositionState,
    plan_rebalance,
    dry_run_plan,
)
from quant_engine.execution.target import TargetPortfolio


def _target(asset="ETH", spot_qty=1.0, perp_qty=-1.0, spot_price=2000.0, perp_price=2000.0):
    """A minimal TargetPortfolio for planner tests (risk fields are placeholders)."""
    return TargetPortfolio(
        asset=asset, equity=spot_qty * spot_price / 0.9, alpha=0.1, sigma_h=0.01,
        spot_price=spot_price, perp_price=perp_price,
        spot_notional=spot_qty * spot_price, perp_notional=spot_qty * perp_price,
        perp_margin=0.0, spot_qty=spot_qty, perp_qty=perp_qty,
        r_liq=1.1, barrier_move_pct=0.1, pi_liq=0.009,
    )


def _guard(max_notional=1e9, max_position=1e9):
    return PreTradeGuard(
        allowed_instruments={"ETH-SPOT", "ETH-PERP", "BTC-SPOT", "BTC-PERP"},
        max_notional_usd=max_notional,
        max_position_qty=max_position,
    )


def test_opens_both_legs_from_flat():
    plan = plan_rebalance(_target(spot_qty=1.0, perp_qty=-1.0), PositionState())
    assert len(plan.orders) == 2
    by_inst = {o.instrument: o for o in plan.orders}
    assert by_inst["ETH-SPOT"].side == "buy"
    assert by_inst["ETH-PERP"].side == "sell"     # opening the short
    assert by_inst["ETH-SPOT"].qty == pytest.approx(1.0)
    assert by_inst["ETH-PERP"].qty == pytest.approx(1.0)


def test_orders_are_delta_neutral_from_flat():
    plan = plan_rebalance(_target(spot_qty=2.5, perp_qty=-2.5), PositionState())
    spot = next(o for o in plan.orders if o.instrument == "ETH-SPOT")
    perp = next(o for o in plan.orders if o.instrument == "ETH-PERP")
    assert spot.qty == pytest.approx(perp.qty)


def test_noop_when_already_at_target():
    tgt = _target(spot_qty=1.0, perp_qty=-1.0)
    plan = plan_rebalance(tgt, PositionState(spot_qty=1.0, perp_qty=-1.0))
    assert plan.is_noop
    assert plan.orders == []
    assert set(plan.skipped) == {"spot", "perp"}


def test_skips_dust_legs():
    tgt = _target(spot_qty=1.0, perp_qty=-1.0)
    # current is within $5 of target on both legs (< $10 min notional)
    cur = PositionState(spot_qty=1.0 - 0.002, perp_qty=-1.0 + 0.002)
    plan = plan_rebalance(tgt, cur, min_trade_notional=10.0)
    assert plan.is_noop


def test_reduce_position_flips_sides():
    tgt = _target(spot_qty=1.0, perp_qty=-1.0)
    cur = PositionState(spot_qty=2.0, perp_qty=-2.0)  # over-exposed
    plan = plan_rebalance(tgt, cur)
    by_inst = {o.instrument: o for o in plan.orders}
    assert by_inst["ETH-SPOT"].side == "sell"     # sell down the long
    assert by_inst["ETH-PERP"].side == "buy"      # buy back part of the short
    assert by_inst["ETH-SPOT"].qty == pytest.approx(1.0)


def test_perp_side_grows_short_when_more_negative():
    # target wants a bigger short than current
    tgt = _target(spot_qty=1.0, perp_qty=-3.0)
    cur = PositionState(spot_qty=1.0, perp_qty=-1.0)
    plan = plan_rebalance(tgt, cur)
    perp = next(o for o in plan.orders if o.instrument == "ETH-PERP")
    assert perp.side == "sell"
    assert perp.qty == pytest.approx(2.0)


def test_dry_run_reaches_target():
    tgt = _target(spot_qty=1.0, perp_qty=-1.0, spot_price=2000.0, perp_price=2000.0)
    broker = PaperBroker(guard=_guard(), positions={})  # seeded flat
    plan = plan_rebalance(tgt, PositionState())
    dry_run_plan(plan, broker)
    assert broker.position("ETH-SPOT") == pytest.approx(1.0)
    assert broker.position("ETH-PERP") == pytest.approx(-1.0)


def test_dry_run_from_existing_position_reaches_target():
    tgt = _target(spot_qty=1.0, perp_qty=-1.0)
    broker = PaperBroker(guard=_guard(), positions={"ETH-SPOT": 2.0, "ETH-PERP": -2.0})
    plan = plan_rebalance(tgt, PositionState(spot_qty=2.0, perp_qty=-2.0))
    dry_run_plan(plan, broker)
    assert broker.position("ETH-SPOT") == pytest.approx(1.0)
    assert broker.position("ETH-PERP") == pytest.approx(-1.0)


def test_dry_run_guard_rejects_oversized_order():
    tgt = _target(spot_qty=100.0, perp_qty=-100.0, spot_price=2000.0)
    broker = PaperBroker(guard=_guard(max_notional=1000.0), positions={})
    plan = plan_rebalance(tgt, PositionState())
    with pytest.raises(GuardRejection):
        dry_run_plan(plan, broker)
