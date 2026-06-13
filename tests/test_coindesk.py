import pytest

from quant_engine.data.coindesk import CoinDeskClient


def test_parse_funding_normalizes_and_sorts():
    payload = {
        "Data": [
            {"TIMESTAMP": 1780916400, "MARKET": "binance", "CLOSE": 0.00000649, "INTERVAL_MS": 28800000},
            {"TIMESTAMP": 1780912800, "MARKET": "binance", "CLOSE": 0.00001313, "INTERVAL_MS": 28800000},
        ],
        "Err": {},
    }
    df = CoinDeskClient._parse_funding(payload)
    assert list(df["ts"]) == [1780912800000, 1780916400000]  # sorted ascending, ms
    assert df["interval_hours"].iloc[0] == 8.0  # 28_800_000 ms / 3.6e6
    assert df["funding_rate"].iloc[-1] == pytest.approx(0.00000649)


def test_parse_funding_raises_on_err():
    with pytest.raises(RuntimeError):
        CoinDeskClient._parse_funding({"Data": [], "Err": {"message": "bad request"}})


def test_parse_funding_raises_on_empty():
    with pytest.raises(ValueError):
        CoinDeskClient._parse_funding({"Data": [], "Err": {}})
