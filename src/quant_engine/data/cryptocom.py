"""Crypto.com Exchange public REST client for perpetual market data.

Public market-data endpoints require no authentication. This client fetches
perpetual OHLCV candles and funding-rate history, normalizes them into pandas
DataFrames aligned on timestamp, and optionally caches to CSV so backtests are
reproducible offline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.crypto.com/exchange/v1"
DEFAULT_TIMEOUT = 15


class CryptoComClient:
    """Thin wrapper over Crypto.com public market-data endpoints."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        cache_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()

    def _get(self, method: str, params: dict) -> dict:
        url = f"{self.base_url}/public/{method}"
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        code = payload.get("code", 0)
        if code != 0:
            raise RuntimeError(
                f"Crypto.com API error for {method}: code={code} {payload.get('message')}"
            )
        return payload["result"]

    def fetch_candles(self, instrument: str, timeframe: str = "1h", count: int = 300) -> pd.DataFrame:
        """Return OHLCV candles sorted ascending by time."""
        result = self._get(
            "get-candlestick",
            {"instrument_name": instrument, "timeframe": timeframe, "count": count},
        )
        df = pd.DataFrame(result.get("data", []))
        if df.empty:
            raise ValueError(f"No candle data returned for {instrument} {timeframe}")
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "ts"}
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        return df[["ts", "time", "open", "high", "low", "close", "volume"]]

    def fetch_funding(self, instrument: str, count: int = 300) -> pd.DataFrame:
        """Return funding-rate history sorted ascending by time."""
        result = self._get(
            "get-valuations",
            {"instrument_name": instrument, "valuation_type": "funding_hist", "count": count},
        )
        df = pd.DataFrame(result.get("data", []))
        if df.empty:
            raise ValueError(f"No funding data returned for {instrument}")
        df = df.rename(columns={"v": "funding_rate", "t": "ts"})
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)
        return df[["ts", "funding_rate"]]

    def load_market(
        self,
        instrument: str,
        timeframe: str = "1h",
        count: int = 300,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch candles + funding, merge on timestamp, cache to CSV when enabled.

        Funding rate is forward-filled onto the candle grid; bars with no funding
        observation default to 0.0 so the backtest never sees NaN.
        """
        cache_path = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{instrument}_{timeframe}_{count}.csv"
            if use_cache and cache_path.exists():
                return pd.read_csv(cache_path, parse_dates=["time"])

        candles = self.fetch_candles(instrument, timeframe, count)
        funding = self.fetch_funding(instrument, count)
        merged = candles.merge(funding, on="ts", how="left")
        merged["funding_rate"] = merged["funding_rate"].ffill().fillna(0.0)

        if cache_path is not None:
            merged.to_csv(cache_path, index=False)
        return merged

    def load_carry_market(
        self,
        perp_instrument: str,
        spot_instrument: str,
        timeframe: str = "1h",
        count: int = 300,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch perp candles + funding and spot candles, aligned on timestamp.

        Returns columns: ts, time, perp_close, spot_close, funding_rate. The spot
        leg is the price hedge for a market-neutral funding carry. Only bars
        present on both the perp and spot grids are kept (inner join).
        """
        cache_path = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = (
                self.cache_dir
                / f"carry_{perp_instrument}_{spot_instrument}_{timeframe}_{count}.csv"
            )
            if use_cache and cache_path.exists():
                return pd.read_csv(cache_path, parse_dates=["time"])

        perp = self.fetch_candles(perp_instrument, timeframe, count)[["ts", "time", "close"]]
        perp = perp.rename(columns={"close": "perp_close"})
        funding = self.fetch_funding(perp_instrument, count)
        spot = self.fetch_candles(spot_instrument, timeframe, count)[["ts", "close"]]
        spot = spot.rename(columns={"close": "spot_close"})

        merged = perp.merge(funding, on="ts", how="left").merge(spot, on="ts", how="inner")
        merged["funding_rate"] = merged["funding_rate"].ffill().fillna(0.0)
        merged = merged.sort_values("ts").reset_index(drop=True)
        merged = merged[["ts", "time", "perp_close", "spot_close", "funding_rate"]]

        if cache_path is not None:
            merged.to_csv(cache_path, index=False)
        return merged
