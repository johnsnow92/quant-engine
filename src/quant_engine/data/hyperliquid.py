"""Hyperliquid public info API client — native funding for the carry validation.

The cross-venue hunt flagged Hyperliquid as carrying persistently rich funding
(13-45%/yr on several alts). This client pulls Hyperliquid's OWN hourly funding
history (no auth) in correct units — ``fundingRate`` is the per-hour rate as a
decimal fraction — so the carry backtest runs on real, properly-scaled data
(unlike Coinalyze, which mis-ingests some venues' funding units).

The ``fundingHistory`` endpoint returns at most ~500 records per call, so deep
history is walked by advancing ``startTime`` until now.
"""
from __future__ import annotations

import pandas as pd
import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_TIMEOUT = 20
HOUR_MS = 3_600_000


class HyperliquidClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def _post(self, body: dict) -> list | dict:
        resp = self.session.post(INFO_URL, json=body, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def fetch_funding_history(self, coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Hourly funding history for ``coin`` in [start_ms, end_ms].

        Paginates the 500-record cap by walking startTime forward. Columns:
        ts(ms), funding_rate (per-hour fraction).
        """
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._post({"type": "fundingHistory", "coin": coin,
                                "startTime": cursor, "endTime": end_ms})
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1]["time"]
            if last <= cursor:  # no forward progress -> done
                break
            cursor = last + 1
            if len(batch) < 500:  # last page
                break
        if not rows:
            raise ValueError(f"No Hyperliquid funding history for {coin}")
        df = pd.DataFrame(rows).drop_duplicates(subset="time")
        df["ts"] = pd.to_numeric(df["time"], errors="coerce")
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)

    def fetch_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """OHLCV candles for ``coin`` in [start_ms, end_ms].

        Paginates by advancing startTime on the candle ``T`` (close-time) field.
        Columns: ts(ms), open, high, low, close, volume.
        """
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._post({
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval,
                        "startTime": cursor, "endTime": end_ms},
            })
            if not batch:
                break
            rows.extend(batch)
            last_t = batch[-1]["T"]
            if last_t <= cursor:
                break
            cursor = last_t + 1
            if len(batch) < 5000:
                break
        if not rows:
            raise ValueError(f"No Hyperliquid candle data for {coin}")
        df = pd.DataFrame(rows).drop_duplicates(subset="T")
        df["ts"] = pd.to_numeric(df["T"], errors="coerce")
        for col, src in [("open", "o"), ("high", "h"), ("low", "l"),
                         ("close", "c"), ("volume", "v")]:
            df[col] = pd.to_numeric(df[src], errors="coerce")
        return (
            df[["ts", "open", "high", "low", "close", "volume"]]
            .sort_values("ts")
            .reset_index(drop=True)
        )

    def current_funding(self) -> pd.DataFrame:
        """Current hourly funding for the whole perp universe. Columns: coin, funding_rate."""
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        names = [a["name"] for a in meta["universe"]]
        rows = [{"coin": n, "funding_rate": float(c.get("funding", "nan"))}
                for n, c in zip(names, ctxs)]
        return pd.DataFrame(rows)
