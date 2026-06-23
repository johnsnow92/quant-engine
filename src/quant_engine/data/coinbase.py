"""Coinbase Exchange public market-data client — deep spot OHLCV history.

Hyperliquid's candle endpoint only serves ~200 days of hourly bars, too short to
backtest the full funding regime. Coinbase Exchange's public candles API needs no
auth, is US-accessible, and serves years of hourly spot OHLCV. It is used to
supply the spot-leg price series for the carry backtest while funding still comes
from Hyperliquid (the venue actually paying it).

The endpoint caps each response at 300 candles, so deep history is walked in
fixed time windows. Candle row format is [time(s), low, high, open, close,
volume], newest-first; this client normalizes to the same schema as
HyperliquidClient.fetch_candles: ts(ms), open, high, low, close, volume.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.exchange.coinbase.com"
DEFAULT_TIMEOUT = 20
HOUR_S = 3600
# Coinbase caps responses at 300 candles; chunk below that to avoid the
# "granularity too small for the requested time range" error at the boundary.
CHUNK_CANDLES = 290


class CoinbaseClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "quant-engine/1.0")

    def _get_candles(self, product_id: str, start_s: int, end_s: int, granularity: int) -> list:
        url = f"{BASE_URL}/products/{product_id}/candles"
        params = {"granularity": granularity, "start": start_s, "end": end_s}
        for attempt in range(5):
            resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429:  # public rate limit — back off and retry
                time.sleep(1.0 + attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return []

    def fetch_candles(
        self,
        product_id: str,
        start_ms: int,
        end_ms: int,
        granularity: int = HOUR_S,
        pace_seconds: float = 0.2,
    ) -> pd.DataFrame:
        """Spot OHLCV for ``product_id`` (e.g. 'ETH-USD') in [start_ms, end_ms].

        Paginates Coinbase's 300-candle cap by walking forward in time windows.
        Overlapping boundary candles are de-duplicated. Columns: ts(ms), open,
        high, low, close, volume.
        """
        start_s = start_ms // 1000
        end_s = end_ms // 1000
        window_s = CHUNK_CANDLES * granularity
        rows: list = []
        cursor = start_s
        while cursor < end_s:
            win_end = min(cursor + window_s, end_s)
            batch = self._get_candles(product_id, cursor, win_end, granularity)
            if batch:
                rows.extend(batch)
            cursor = win_end  # 1-candle overlap, removed by drop_duplicates
            if pace_seconds:
                time.sleep(pace_seconds)
        if not rows:
            raise ValueError(f"No Coinbase candle data for {product_id}")
        df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
        df = df.drop_duplicates(subset="t")
        df["ts"] = pd.to_numeric(df["t"], errors="coerce") * 1000  # seconds -> ms
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return (
            df[["ts", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["ts", "close"])
            .sort_values("ts")
            .reset_index(drop=True)
        )
