"""Tests for the Coinbase Exchange public candles client (data/coinbase.py).

Network is faked so the tests exercise pagination, schema normalization
(seconds→ms, Coinbase's [time,low,high,open,close,volume] order), boundary
de-duplication, and ascending sort without hitting the API.
"""
import pytest

from quant_engine.data.coinbase import CoinbaseClient, CHUNK_CANDLES

GRAN = 3600
WINDOW_S = CHUNK_CANDLES * GRAN


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Returns a queued payload per .get() call; records params."""
    def __init__(self, batches):
        self.batches = batches
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        idx = len(self.calls)
        self.calls.append(params)
        payload = self.batches[idx] if idx < len(self.batches) else []
        return _FakeResp(payload)


def _client(batches):
    return CoinbaseClient(session=_FakeSession(batches))


def test_normalizes_schema_and_sorts_ascending():
    # Coinbase rows are newest-first: [time(s), low, high, open, close, volume]
    batch = [
        [3600, 10.0, 12.0, 11.0, 11.5, 100.0],  # newer
        [0, 9.0, 11.0, 10.0, 10.5, 90.0],        # older
    ]
    c = _client([batch])
    df = c.fetch_candles("ETH-USD", 0, 2 * GRAN * 1000, pace_seconds=0)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert list(df["ts"]) == [0, 3_600_000]          # ascending, seconds→ms
    assert df["close"].iloc[0] == pytest.approx(10.5)
    assert df["open"].iloc[1] == pytest.approx(11.0)
    assert df["high"].iloc[1] == pytest.approx(12.0)
    assert df["low"].iloc[0] == pytest.approx(9.0)


def test_paginates_across_windows():
    b1 = [[0, 1, 1, 1, 1, 1]]
    b2 = [[WINDOW_S, 2, 2, 2, 2, 2]]
    c = _client([b1, b2])
    end_ms = 2 * WINDOW_S * 1000
    df = c.fetch_candles("BTC-USD", 0, end_ms, pace_seconds=0)
    assert len(c.session.calls) == 2          # two time windows walked
    assert len(df) == 2


def test_dedups_boundary_overlap():
    # Same timestamp returned at the window boundary in both batches.
    b1 = [[WINDOW_S, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]
    b2 = [[WINDOW_S, 1, 1, 1, 1, 1]]            # duplicate of b1's first row
    c = _client([b1, b2])
    df = c.fetch_candles("ETH-USD", 0, 2 * WINDOW_S * 1000, pace_seconds=0)
    assert df["ts"].is_unique
    assert len(df) == 2


def test_raises_when_no_data():
    c = _client([[]])
    with pytest.raises(ValueError, match="No Coinbase candle data"):
        c.fetch_candles("ETH-USD", 0, GRAN * 1000, pace_seconds=0)


def test_request_params_use_granularity_and_window():
    c = _client([[[0, 1, 1, 1, 1, 1]]])
    c.fetch_candles("ETH-USD", 0, GRAN * 1000, granularity=GRAN, pace_seconds=0)
    params = c.session.calls[0]
    assert params["granularity"] == GRAN
    assert params["start"] == 0
