"""Unit tests for CoinbaseCfmReader — all HTTP mocked. spec docs/plans/07 §4.1.

Read-only adapter: proves each CFM payload normalizes to the safety-stack type
the reconciler / gates / funding-check consume, and that a leg snapshot feeds
the already-built reconcile() cleanly.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_engine.execution.coinbase_cfm_broker import (
    VENUE,
    CfmDataError,
    CoinbaseCfmReader,
    _margin_buffer_pct,
    _signed_size,
)
from quant_engine.execution.reconcile import (
    LegSnapshot,
    PositionSnapshot,
    net_delta,
    reconcile,
)


def _make_reader() -> CoinbaseCfmReader:
    """Build a reader with a mocked RESTClient — no real CFM credentials."""
    env = {
        "COINBASE_CFM_API_KEY_NAME": "organizations/test/apiKeys/cfm1",
        "COINBASE_CFM_PRIVATE_KEY": "fake-key",
    }
    mock_client = MagicMock()
    with patch.dict("os.environ", env):
        with patch(
            "quant_engine.execution.coinbase_cfm_broker.RESTClient",
            return_value=mock_client,
        ):
            reader = CoinbaseCfmReader()
    reader._client = mock_client
    return reader


# ---------------------------------------------------------------------------
# Credentials / construction
# ---------------------------------------------------------------------------

def test_requires_cfm_credentials():
    with patch.dict("os.environ", {}, clear=True):
        with patch("quant_engine.execution.coinbase_cfm_broker.RESTClient"):
            with pytest.raises(EnvironmentError, match="COINBASE_CFM"):
                CoinbaseCfmReader()


# ---------------------------------------------------------------------------
# Position normalization (the short leg is the whole point)
# ---------------------------------------------------------------------------

def test_position_btc_from_net_size():
    reader = _make_reader()
    reader._client.get_futures_position.return_value = {"position": {"net_size": "-0.04"}}
    assert reader.position_btc("BTCUSD-PERP") == pytest.approx(-0.04)
    assert reader._client.get_futures_position.call_args.kwargs["product_id"] == "BTC-PERP"


def test_position_btc_from_contracts_and_side():
    reader = _make_reader()
    # 4 nano BTC contracts (0.01 each) short → -0.04 BTC
    reader._client.get_futures_position.return_value = {
        "position": {"number_of_contracts": "4", "side": "SHORT"}
    }
    assert reader.position_btc("BTCUSD-PERP") == pytest.approx(-0.04)


def test_position_long_side_is_positive():
    assert _signed_size({"number_of_contracts": "4", "side": "LONG"}, 0.01) == pytest.approx(0.04)


def test_position_unparseable_raises():
    assert pytest.raises(CfmDataError, _signed_size, {"foo": "bar"}, 0.01)


def test_position_unmapped_instrument_raises():
    reader = _make_reader()
    with pytest.raises(CfmDataError, match="no CFM product mapping"):
        reader.position_btc("DOGEUSD-PERP")


# ---------------------------------------------------------------------------
# Margin buffer (the Coinbase 4pm step-up shows up here)
# ---------------------------------------------------------------------------

def test_margin_buffer_explicit_field():
    reader = _make_reader()
    reader._client.get_futures_balance_summary.return_value = {
        "balance_summary": {"liquidation_buffer_percentage": "30"}
    }
    assert reader.margin_buffer_pct() == pytest.approx(30.0)


def test_margin_buffer_derived_from_equity_vs_threshold():
    # equity 1300, liquidation threshold 1000 → 30% above
    summary = {
        "total_balance": {"value": "1300"},
        "liquidation_threshold": {"value": "1000"},
    }
    assert _margin_buffer_pct(summary) == pytest.approx(30.0)


def test_margin_buffer_unparseable_raises():
    assert pytest.raises(CfmDataError, _margin_buffer_pct, {})


# ---------------------------------------------------------------------------
# Mark price + funding rate (feeds edge gate + funding-convention check)
# ---------------------------------------------------------------------------

def test_mark_price():
    reader = _make_reader()
    reader._client.get_product.return_value = {"product": {"price": "63000.0"}}
    assert reader.mark_price("BTCUSD-PERP") == pytest.approx(63_000.0)


def test_funding_rate_annualized_from_hourly():
    reader = _make_reader()
    # Hourly rate that annualizes (×8760) to ≈ +6%/yr — the carry thesis figure.
    reader._client.get_product.return_value = {
        "product": {"future_product_details": {"funding_rate": "0.00000685"}}
    }
    assert reader.funding_rate_annual("BTCUSD-PERP") == pytest.approx(0.06, abs=2e-3)


def test_funding_rate_negative_sign_preserved():
    reader = _make_reader()
    reader._client.get_product.return_value = {
        "product": {"perpetual_details": {"funding_rate": "-0.00000685"}}
    }
    assert reader.funding_rate_annual("BTCUSD-PERP") < 0.0


def test_funding_rate_missing_raises():
    reader = _make_reader()
    reader._client.get_product.return_value = {"product": {}}
    with pytest.raises(CfmDataError, match="no funding rate"):
        reader.funding_rate_annual("BTCUSD-PERP")


# ---------------------------------------------------------------------------
# The bridge: adapter output flows into the already-built reconcile()
# ---------------------------------------------------------------------------

def test_leg_snapshot_feeds_reconcile():
    reader = _make_reader()
    reader._client.get_futures_position.return_value = {"position": {"net_size": "-0.04"}}
    reader._client.get_futures_balance_summary.return_value = {
        "balance_summary": {"liquidation_buffer_percentage": "30"}
    }

    short_leg = reader.leg_snapshot("BTCUSD-PERP")
    assert isinstance(short_leg, LegSnapshot)
    assert short_leg.venue == VENUE
    assert short_leg.position_btc == pytest.approx(-0.04)
    assert short_leg.margin_buffer_pct == pytest.approx(30.0)

    # Paired with a synthetic long Kalshi leg, the position is delta-neutral and
    # the reconciler reports healthy — the adapter is a drop-in recon input.
    snap = PositionSnapshot(
        long=LegSnapshot("kalshi-perp", 0.04, 30.0),
        short=short_leg,
    )
    assert net_delta(snap) == pytest.approx(0.0)
    result = reconcile(snap)
    assert result.ok
    assert result.should_flatten is False


def test_unknown_side_raises():
    # An unrecognized side must NOT be silently treated as long (wrong sign to recon).
    assert pytest.raises(CfmDataError, _signed_size, {"number_of_contracts": "4", "side": "WEIRD"}, 0.01)


def test_missing_side_raises():
    assert pytest.raises(CfmDataError, _signed_size, {"number_of_contracts": "4"}, 0.01)
