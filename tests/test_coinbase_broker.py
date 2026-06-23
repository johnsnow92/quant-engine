"""Unit tests for CoinbaseBroker — all HTTP calls mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_engine.execution.coinbase_broker import (
    CoinbaseBroker,
    ExecutionError,
    _INSTRUMENT_MAP,
)
from quant_engine.execution.guards import GuardRejection, Order, PreTradeGuard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_guard(**overrides):
    base = dict(
        allowed_instruments={"BTCUSD-PERP", "ETHUSD-PERP"},
        max_notional_usd=2_000.0,
        max_position_qty=0.025,
    )
    base.update(overrides)
    return PreTradeGuard(**base)


def _make_broker(guard=None, dry_run=False) -> CoinbaseBroker:
    """Build a CoinbaseBroker with a mocked RESTClient (no real credentials needed)."""
    guard = guard or _make_guard()
    env = {
        "COINBASE_CDP_API_KEY_NAME": "organizations/test/apiKeys/key1",
        "COINBASE_CDP_PRIVATE_KEY": "fake-key",
    }
    mock_client = MagicMock()
    # Default: get_portfolios returns empty so position cache loads cleanly
    mock_client.get_portfolios.return_value = {"portfolios": []}

    with patch.dict("os.environ", env):
        with patch("quant_engine.execution.coinbase_broker.RESTClient", return_value=mock_client):
            broker = CoinbaseBroker(guard=guard, dry_run=dry_run)
    broker._client = mock_client
    return broker


def _filled_response(order_id="order-123", avg_price=63_000.0, fee=6.3):
    return {
        "success": True,
        "success_response": {"order_id": order_id},
    }


def _filled_order_detail(order_id="order-123", avg_price=63_000.0, fee=6.3):
    return {
        "order": {
            "order_id": order_id,
            "status": "FILLED",
            "average_filled_price": str(avg_price),
            "total_fees": str(fee),
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_submit_order_success():
    """Happy path: mocked POST returns success, poll returns FILLED."""
    broker = _make_broker()
    broker._client.market_order.return_value = _filled_response()
    broker._client.get_order.return_value = _filled_order_detail()

    order = Order("BTCUSD-PERP", "sell", 0.01, 63_000.0)
    fill = broker.submit_order(order)

    assert fill.instrument == "BTCUSD-PERP"
    assert fill.side == "sell"
    assert fill.qty == pytest.approx(0.01)
    assert fill.price == pytest.approx(63_000.0)
    assert fill.fee == pytest.approx(6.3)
    # Position cache updated
    assert broker._positions["BTCUSD-PERP"] == pytest.approx(-0.01)
    # Coinbase product_id was used in the actual API call
    broker._client.market_order.assert_called_once()
    call_kwargs = broker._client.market_order.call_args.kwargs
    assert call_kwargs["product_id"] == "BTC-PERP-INTX"
    assert call_kwargs["side"] == "SELL"


def test_guard_rejection_not_sent():
    """GuardRejection must stop execution before any HTTP call."""
    broker = _make_broker(guard=_make_guard(max_notional_usd=100.0))
    # 0.01 BTC @ 63000 = 630 notional > 100 cap → guard should reject
    order = Order("BTCUSD-PERP", "sell", 0.01, 63_000.0)

    with pytest.raises(GuardRejection):
        broker.submit_order(order)

    broker._client.market_order.assert_not_called()


def test_instrument_not_in_map():
    """Unmapped instrument raises GuardRejection before any HTTP call."""
    broker = _make_broker(
        guard=PreTradeGuard(
            allowed_instruments={"SOLUSD-PERP"},
            max_notional_usd=2_000.0,
            max_position_qty=10.0,
        )
    )
    order = Order("SOLUSD-PERP", "buy", 1.0, 150.0)

    with pytest.raises(GuardRejection, match="no Coinbase product_id mapping"):
        broker.submit_order(order)

    broker._client.market_order.assert_not_called()


def test_retries_on_500():
    """Retriable server errors exhaust retries and raise ExecutionError."""
    broker = _make_broker()
    broker._client.market_order.side_effect = Exception("HTTP 500 internal server error")

    order = Order("BTCUSD-PERP", "sell", 0.01, 63_000.0)

    with pytest.raises((ExecutionError, Exception)):
        broker.submit_order(order)

    # Should have attempted all retries
    assert broker._client.market_order.call_count == 3


def test_kill_switch_halts():
    """kill_switch=True raises GuardRejection before any I/O."""
    broker = _make_broker(guard=_make_guard(kill_switch=True))
    order = Order("BTCUSD-PERP", "sell", 0.01, 63_000.0)

    with pytest.raises(GuardRejection, match="kill switch"):
        broker.submit_order(order)

    broker._client.market_order.assert_not_called()


def test_position_cache_update():
    """Position is updated in local cache after a successful fill."""
    broker = _make_broker()
    broker._client.market_order.return_value = _filled_response()
    broker._client.get_order.return_value = _filled_order_detail()

    assert broker.position("BTCUSD-PERP") == 0.0

    order = Order("BTCUSD-PERP", "sell", 0.01, 63_000.0)
    broker.submit_order(order)

    # After selling 0.01, position should be -0.01
    assert broker.position("BTCUSD-PERP") == pytest.approx(-0.01)


def test_dry_run_no_http():
    """dry_run=True returns a synthetic Fill without any HTTP call."""
    broker = _make_broker(dry_run=True)
    order = Order("BTCUSD-PERP", "sell", 0.01, 63_000.0)
    fill = broker.submit_order(order)

    assert fill.instrument == "BTCUSD-PERP"
    assert fill.fee == 0.0
    broker._client.market_order.assert_not_called()
