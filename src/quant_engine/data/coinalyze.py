"""Coinalyze REST client — cross-venue funding-rate breadth layer.

Coinalyze aggregates perpetual funding rates across 30+ venues for free
(40 req/min per key). It complements the engine's primary funding sources
(Kraken Futures + CoinDesk MCP) by answering "is Kraken's funding rich or poor
vs the rest of the market, and where is carry paying best right now?" — the
selection layer for a Kraken-executed carry book.

Requires a free API key: register at https://coinalyze.net and set
``COINALYZE_API_KEY`` (or pass ``api_key``). Without it, calls raise a clear
error rather than silently returning nothing.

Symbol format is ``<BASE><QUOTE>_PERP.<X>`` where ``<X>`` is a one-letter
exchange code (resolve via ``fetch_exchanges``). Funding rates are returned as
per-interval decimal fractions, matching the carry backtest's units.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.coinalyze.net/v1"
DEFAULT_TIMEOUT = 20


class CoinalyzeError(RuntimeError):
    """Raised on a missing key or a non-OK Coinalyze response."""


class CoinalyzeClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("COINALYZE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        if not self.api_key:
            raise CoinalyzeError(
                "COINALYZE_API_KEY not set. Register free at coinalyze.net and "
                "export COINALYZE_API_KEY=<key> to use the cross-venue funding layer."
            )
        # Coinalyze caps at 40 req/min; back off and retry on 429 (window resets
        # within ~60s) rather than aborting a long scan mid-flight.
        for attempt in range(4):
            resp = self.session.get(
                f"{self.base_url}/{path}",
                params=params or {},
                headers={"api_key": self.api_key},
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 429:
                if attempt == 3:
                    raise CoinalyzeError("Coinalyze rate limit: still 429 after backoff.")
                time.sleep(20 * (attempt + 1))  # 20s, 40s, 60s
                continue
            resp.raise_for_status()
            return resp.json()

    def fetch_future_markets(self) -> pd.DataFrame:
        """All perpetual/future markets across venues. Columns include symbol, base_asset, exchange."""
        data = self._get("future-markets")
        df = pd.DataFrame(data)
        return df

    def fetch_current_funding(self, symbols: list[str]) -> pd.DataFrame:
        """Latest funding rate per symbol. Columns: symbol, funding_rate (fraction)."""
        data = self._get("funding-rate", {"symbols": ",".join(symbols)})
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df = df.rename(columns={"value": "funding_rate"})
        return df[["symbol", "funding_rate"]]

    def fetch_exchanges(self) -> dict[str, str]:
        """Map of exchange code -> human name (e.g. 'K' -> 'Kraken')."""
        data = self._get("exchanges")
        return {row["code"]: row["name"] for row in data}

    def fetch_funding_history(
        self, symbols: list[str], interval: str = "daily", frm: int = 0, to: int = 0
    ) -> pd.DataFrame:
        """Funding-rate OHLC history. Returns long frame: symbol, t(s), funding_rate(=close)."""
        data = self._get(
            "funding-rate-history",
            {"symbols": ",".join(symbols), "interval": interval, "from": frm, "to": to},
        )
        rows = []
        for series in data:
            sym = series["symbol"]
            for pt in series.get("history", []):
                rows.append({"symbol": sym, "t": pt["t"], "funding_rate": pt["c"]})
        return pd.DataFrame(rows)

    def cross_venue_funding(self, base: str = "BTC") -> pd.DataFrame:
        """Current funding for every perp on ``base`` across venues, richest first.

        The selection scanner: shows which venue pays the most to short (or long)
        the perp, contextualizing Kraken's own funding within the market.
        """
        markets = self.fetch_future_markets()
        if markets.empty:
            return markets
        perps = markets[
            (markets.get("base_asset", "").astype(str).str.upper() == base.upper())
            & (markets.get("is_perpetual", True) == True)  # noqa: E712
        ]
        symbols = perps["symbol"].tolist()
        if not symbols:
            return pd.DataFrame(columns=["symbol", "funding_rate"])
        funding = self.fetch_current_funding(symbols)
        return funding.sort_values("funding_rate", ascending=False).reset_index(drop=True)
