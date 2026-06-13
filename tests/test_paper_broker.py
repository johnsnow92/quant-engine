import pytest

from quant_engine.execution.guards import GuardRejection, Order, PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker


def make_broker(**overrides):
    guard = PreTradeGuard(
        allowed_instruments={"BTCUSD-PERP"},
        max_notional_usd=1_000_000,
        max_position_qty=10.0,
    )
    return PaperBroker(guard=guard, cash=100_000, **overrides)


def test_buy_updates_position_and_cash():
    broker = make_broker(fee_bps=0.0)
    broker.submit_order(Order("BTCUSD-PERP", "buy", 1.0, 50_000))
    assert broker.position("BTCUSD-PERP") == 1.0
    assert broker.cash == pytest.approx(50_000)


def test_buy_then_sell_reconciles():
    broker = make_broker(fee_bps=0.0)
    broker.submit_order(Order("BTCUSD-PERP", "buy", 2.0, 50_000))
    broker.submit_order(Order("BTCUSD-PERP", "sell", 0.5, 55_000))
    assert broker.position("BTCUSD-PERP") == pytest.approx(1.5)
    assert broker.cash == pytest.approx(27_500)  # 100k -100k +27.5k
    assert len(broker.fills) == 2


def test_fee_deducted_on_notional():
    broker = make_broker(fee_bps=10.0)  # 10 bps
    broker.submit_order(Order("BTCUSD-PERP", "buy", 1.0, 50_000))
    assert broker.cash == pytest.approx(49_950)  # 100k -50k -50 fee


def test_guard_rejection_prevents_fill():
    guard = PreTradeGuard(
        allowed_instruments={"BTCUSD-PERP"}, max_notional_usd=10_000, max_position_qty=10.0
    )
    broker = PaperBroker(guard=guard, cash=100_000)
    with pytest.raises(GuardRejection):
        broker.submit_order(Order("BTCUSD-PERP", "buy", 1.0, 50_000))
    assert broker.position("BTCUSD-PERP") == 0.0
    assert broker.cash == 100_000
    assert len(broker.fills) == 0


def test_equity_marks_to_market():
    broker = make_broker(fee_bps=0.0)
    broker.submit_order(Order("BTCUSD-PERP", "buy", 1.0, 50_000))
    assert broker.equity({"BTCUSD-PERP": 60_000}) == pytest.approx(110_000)
