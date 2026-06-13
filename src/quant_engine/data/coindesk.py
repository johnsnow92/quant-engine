"""CoinDesk (CCData) futures REST client — second-venue funding data.

This is the engine-side twin of the CoinDesk MCP server: same data, reachable
from standalone code via ``data-api.coindesk.com``. Public funding-rate history
works without a key on the free tier; set ``COINDESK_API_KEY`` (or pass
``api_key``) to lift rate limits.

Funding is returned with its native settlement interval (``interval_hours``,
derived from ``INTERVAL_MS``) so callers can normalize venues with different
funding schedules onto a common grid before comparing.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

BASE_URL = "https://data-api.coindesk.com"
DEFAULT_TIMEOUT = 15


class CoinDeskClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("COINDESK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    @staticmethod
    def _parse_funding(payload: dict) -> pd.DataFrame:
        err = payload.get("Err") or {}
        if err:
            raise RuntimeError(f"CoinDesk API error: {err}")
        rows = payload.get("Data", [])
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No funding data returned")
        out = pd.DataFrame(
            {
                "ts": pd.to_numeric(df["TIMESTAMP"], errors="coerce") * 1000,
                "funding_rate": pd.to_numeric(df["CLOSE"], errors="coerce"),
                "interval_hours": pd.to_numeric(df["INTERVAL_MS"], errors="coerce") / 3_600_000.0,
                "market": df.get("MARKET", pd.Series(["?"] * len(df))),
            }
        )
        return out.sort_values("ts").reset_index(drop=True)

    def fetch_funding(
        self,
        market: str,
        instrument: str,
        limit: int = 300,
        frequency: str = "hours",
    ) -> pd.DataFrame:
        """Funding-rate history for ``instrument`` on ``market`` (e.g. binance).

        Returns columns: ts (ms), funding_rate, interval_hours, market.
        """
        url = f"{self.base_url}/futures/v1/historical/funding-rate/{frequency}"
        params = {"market": market, "instrument": instrument, "limit": limit}
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return self._parse_funding(resp.json())
