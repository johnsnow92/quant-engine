"""Unit tests for KalshiPerpReader — all HTTP mocked. spec docs/plans/07 §4.1.

Read-only adapter for the long (Kalshi, 0%-funding) leg. Proves the RSA-PSS
signing is real (generated keypair, verified locally — no network), each payload
normalizes to the safety-stack type, and a leg snapshot feeds reconcile().
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from quant_engine.execution.kalshi_perp_broker import (
    VENUE,
    KalshiDataError,
    KalshiPerpReader,
    _kalshi_margin_buffer_pct,
    _kalshi_mark_price,
    sign_pss,
    signed_message,
)
from quant_engine.execution.reconcile import (
    LegSnapshot,
    PositionSnapshot,
    net_delta,
    reconcile,
)

# One throwaway RSA keypair for the whole module (no real credentials).
_PRIV = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PEM = _PRIV.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("ascii")


def _make_reader(session=None) -> KalshiPerpReader:
    env = {"KALSHI_PERP_KEY_ID": "test-key-id", "KALSHI_PERP_PRIVATE_KEY": _TEST_PEM}
    with patch.dict("os.environ", env):
        return KalshiPerpReader(session=session or MagicMock())


def _resp(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


# ---------------------------------------------------------------------------
# Auth signing (the real, network-free core)
# ---------------------------------------------------------------------------

def test_signed_message_format():
    assert signed_message(1700000000000, "get", "/trade-api/v2/portfolio/positions") == (
        "1700000000000GET/trade-api/v2/portfolio/positions"
    )


def test_sign_pss_produces_verifiable_signature():
    msg = signed_message(1700000000000, "GET", "/trade-api/v2/portfolio/positions")
    sig = base64.b64decode(sign_pss(_TEST_PEM, msg))
    assert len(sig) > 0
    # Verify with the public key — raises InvalidSignature if the scheme is wrong.
    _PRIV.public_key().verify(
        sig,
        msg.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_requires_perp_credentials():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError, match="KALSHI_PERP"):
            KalshiPerpReader()


# ---------------------------------------------------------------------------
# Signed transport wiring
# ---------------------------------------------------------------------------

def test_get_signs_and_targets_correct_url():
    reader = _make_reader()
    reader.session.get.return_value = _resp({"market_positions": []})
    reader.position_btc("BTCUSD-PERP")

    call = reader.session.get.call_args
    assert call.args[0] == "https://external-api.kalshi.com/trade-api/v2/portfolio/positions"
    headers = call.kwargs["headers"]
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-SIGNATURE"]            # non-empty signature
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()


# ---------------------------------------------------------------------------
# Position normalization (the long leg)
# ---------------------------------------------------------------------------

def test_position_btc_long():
    reader = _make_reader()
    # 40 contracts × 0.001 BTC = +0.04 BTC long
    reader.session.get.return_value = _resp(
        {"market_positions": [{"ticker": "BTCUSD-PERP", "position": 40}]}
    )
    assert reader.position_btc("BTCUSD-PERP") == pytest.approx(0.04)


def test_position_btc_short_is_negative():
    reader = _make_reader()
    reader.session.get.return_value = _resp(
        {"market_positions": [{"ticker": "BTCUSD-PERP", "position": -40}]}
    )
    assert reader.position_btc("BTCUSD-PERP") == pytest.approx(-0.04)


def test_position_absent_ticker_is_flat():
    reader = _make_reader()
    reader.session.get.return_value = _resp({"market_positions": []})
    assert reader.position_btc("BTCUSD-PERP") == 0.0


def test_position_unmapped_instrument_raises():
    reader = _make_reader()
    with pytest.raises(KalshiDataError, match="no Kalshi perp mapping"):
        reader.position_btc("DOGEUSD-PERP")


# ---------------------------------------------------------------------------
# Margin buffer
# ---------------------------------------------------------------------------

def test_margin_buffer_explicit():
    assert _kalshi_margin_buffer_pct({"margin_buffer_percentage": "30"}) == pytest.approx(30.0)


def test_margin_buffer_derived():
    summary = {"portfolio_value": "1300", "maintenance_margin": "1000"}
    assert _kalshi_margin_buffer_pct(summary) == pytest.approx(30.0)


def test_margin_buffer_unparseable_raises():
    assert pytest.raises(KalshiDataError, _kalshi_margin_buffer_pct, {})


# ---------------------------------------------------------------------------
# Mark price + funding (dead-band pins funding at ~0)
# ---------------------------------------------------------------------------

def test_mark_price_explicit():
    assert _kalshi_mark_price({"mark_price": "63000"}) == pytest.approx(63_000.0)


def test_mark_price_bid_ask_mid():
    assert _kalshi_mark_price({"yes_bid": "62900", "yes_ask": "63100"}) == pytest.approx(63_000.0)


def test_funding_rate_dead_band_near_zero():
    reader = _make_reader()
    reader.session.get.return_value = _resp({"market": {"funding_rate": "0.0"}})
    assert reader.funding_rate_annual("BTCUSD-PERP") == pytest.approx(0.0)


def test_funding_rate_annualized_sign():
    reader = _make_reader()
    reader.session.get.return_value = _resp({"market": {"funding_rate": "-0.00000685"}})
    assert reader.funding_rate_annual("BTCUSD-PERP") < 0.0


def test_funding_rate_missing_raises():
    reader = _make_reader()
    reader.session.get.return_value = _resp({"market": {}})
    with pytest.raises(KalshiDataError, match="no funding rate"):
        reader.funding_rate_annual("BTCUSD-PERP")


# ---------------------------------------------------------------------------
# The bridge: adapter output flows into the already-built reconcile()
# ---------------------------------------------------------------------------

def test_leg_snapshot_feeds_reconcile():
    reader = _make_reader()
    # position read, then margin read — two different payloads in call order.
    reader.session.get.side_effect = [
        _resp({"market_positions": [{"ticker": "BTCUSD-PERP", "position": 40}]}),
        _resp({"margin": {"margin_buffer_percentage": "30"}}),
    ]

    long_leg = reader.leg_snapshot("BTCUSD-PERP")
    assert isinstance(long_leg, LegSnapshot)
    assert long_leg.venue == VENUE
    assert long_leg.position_btc == pytest.approx(0.04)
    assert long_leg.margin_buffer_pct == pytest.approx(30.0)

    # Paired with a synthetic short CFM leg → delta-neutral, reconciler healthy.
    snap = PositionSnapshot(
        long=long_leg,
        short=LegSnapshot("coinbase-futures", -0.04, 30.0),
    )
    assert net_delta(snap) == pytest.approx(0.0)
    assert reconcile(snap).ok
