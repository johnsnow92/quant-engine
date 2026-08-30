import pytest

from quant_engine.execution.guards import GuardRejection, Order, PreTradeGuard


def make_guard(**overrides):
    base = dict(
        allowed_instruments={"BTCUSD-PERP"},
        max_notional_usd=50_000,
        max_position_qty=2.0,
    )
    base.update(overrides)
    return PreTradeGuard(**base)


def test_valid_order_passes():
    guard = make_guard()
    order = Order("BTCUSD-PERP", "buy", 0.5, 60_000)  # 30k notional
    assert guard.check(order) is order


def test_order_rejects_non_finite_qty():
    with pytest.raises(ValueError, match="qty must be finite"):
        Order("BTCUSD-PERP", "buy", float("nan"), 63_000.0)


def test_order_rejects_non_finite_price():
    with pytest.raises(ValueError, match="price must be finite"):
        Order("BTCUSD-PERP", "buy", 0.04, float("inf"))


def test_rejects_oversized_notional():
    guard = make_guard()
    with pytest.raises(GuardRejection):
        guard.check(Order("BTCUSD-PERP", "buy", 1.0, 60_000))  # 60k > 50k


def test_accepts_order_at_exact_notional_limit():
    guard = make_guard()
    order = Order("BTCUSD-PERP", "buy", 1.0, 50_000)

    assert guard.check(order) is order


def test_rejects_disallowed_instrument():
    guard = make_guard()
    with pytest.raises(GuardRejection):
        guard.check(Order("ETHUSD-PERP", "buy", 0.1, 3_000))


def test_rejects_when_position_limit_breached():
    guard = make_guard()
    with pytest.raises(GuardRejection):
        guard.check(Order("BTCUSD-PERP", "buy", 1.5, 10_000), current_position=1.0)


def test_kill_switch_blocks_everything():
    guard = make_guard(kill_switch=True)
    with pytest.raises(GuardRejection):
        guard.check(Order("BTCUSD-PERP", "buy", 0.01, 60_000))


def test_invalid_order_construction():
    with pytest.raises(ValueError):
        Order("BTCUSD-PERP", "hold", 1, 100)
    with pytest.raises(ValueError):
        Order("BTCUSD-PERP", "buy", -1, 100)
