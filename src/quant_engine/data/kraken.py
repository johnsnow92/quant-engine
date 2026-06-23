"""Kraken public data client for the real-money funding-carry engine.

Supplies the three series the two-leg carry backtest consumes — ``spot_close``,
``perp_close``, ``funding_rate`` — entirely from PUBLIC endpoints (no auth):

  * spot OHLC      -> https://api.kraken.com/0/public/OHLC
  * perp candles   -> https://futures.kraken.com/api/charts/v1/trade/<sym>/<res>
  * funding rates  -> https://futures.kraken.com/derivatives/api/v4/historicalfundingrates

Kraken's perpetual (``PF_*``) funding settles HOURLY; the funding feed returns
``relativeFundingRate`` (the per-hour rate as a decimal fraction, e.g. 1.2e-5)
which is exactly the per-bar funding the carry PnL applies. The absolute
``fundingRate`` field (USD/contract) is deliberately ignored.

Spot OHLC caps at ~720 candles per call, so hourly spot gives ~30 days — enough
to validate the engine end-to-end on real funding. Deeper hourly history is the
documented upgrade path (CoinDesk MCP / paginated walk).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

SPOT_BASE = "https://api.kraken.com/0/public"
FUTURES_BASE = "https://futures.kraken.com"
DEFAULT_TIMEOUT = 20

# Spot pair (altname) + perpetual symbol for each supported asset.
CARRY_SYMBOLS = {
    "BTC": {"spot": "XBTUSD", "perp": "PF_XBTUSD"},
    "ETH": {"spot": "ETHUSD", "perp": "PF_ETHUSD"},
    "SOL": {"spot": "SOLUSD", "perp": "PF_SOLUSD"},
}


class KrakenDataClient:
    """Public spot + futures market data for funding-carry modeling."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()

    # --- spot ---------------------------------------------------------------
    def fetch_spot_ohlc(self, pair: str, interval: int = 60) -> pd.DataFrame:
        """Spot OHLC. interval in minutes (60 = hourly). Columns: ts(ms), close."""
        resp = self.session.get(
            f"{SPOT_BASE}/OHLC", params={"pair": pair, "interval": interval},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken spot OHLC error for {pair}: {payload['error']}")
        result = payload["result"]
        key = next(k for k in result if k != "last")
        rows = result[key]
        if not rows:
            raise ValueError(f"No spot OHLC for {pair}")
        df = pd.DataFrame(rows, columns=[
            "ts", "open", "high", "low", "close", "vwap", "volume", "count"])
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce") * 1000  # -> ms
        df["spot_close"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["ts", "spot_close"]].sort_values("ts").reset_index(drop=True)

    # --- perp candles -------------------------------------------------------
    def fetch_perp_candles(self, symbol: str, resolution: str = "1h") -> pd.DataFrame:
        """Perp trade-price candles. Columns: ts(ms), perp_close."""
        url = f"{FUTURES_BASE}/api/charts/v1/trade/{symbol}/{resolution}"
        resp = self.session.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            raise ValueError(f"No perp candles for {symbol}")
        df = pd.DataFrame(candles)
        df["ts"] = pd.to_numeric(df["time"], errors="coerce")  # already ms
        df["perp_close"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["ts", "perp_close"]].sort_values("ts").reset_index(drop=True)

    # --- funding ------------------------------------------------------------
    def fetch_funding(self, symbol: str) -> pd.DataFrame:
        """Hourly funding history. Columns: ts(ms), funding_rate (per-hour fraction)."""
        url = f"{FUTURES_BASE}/derivatives/api/v4/historicalfundingrates"
        resp = self.session.get(url, params={"symbol": symbol}, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("result") != "success":
            raise RuntimeError(f"Kraken funding error for {symbol}: {payload}")
        rows = payload.get("rates", [])
        if not rows:
            raise ValueError(f"No funding history for {symbol}")
        df = pd.DataFrame(rows)
        dt = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df["ts"] = dt.astype("datetime64[ms]").astype("int64")  # epoch ms, unit-safe
        df["funding_rate"] = pd.to_numeric(df["relativeFundingRate"], errors="coerce")
        return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)

    # --- assembled carry market --------------------------------------------
    def load_carry_market(self, asset: str, spot_interval: int = 60) -> pd.DataFrame:
        """Aligned perp_close / spot_close / funding_rate for one asset.

        Inner-joins the three hourly series on timestamp. If the perp candle feed
        is unavailable, falls back to spot_close for the perp leg (residual basis
        ~0) and tags the frame so the caller knows the basis term is degraded.
        """
        sym = CARRY_SYMBOLS.get(asset.upper())
        if sym is None:
            raise ValueError(f"Unsupported carry asset {asset!r}; add it to CARRY_SYMBOLS")

        spot = self.fetch_spot_ohlc(sym["spot"], spot_interval)
        funding = self.fetch_funding(sym["perp"])
        perp_ok = True
        try:
            perp = self.fetch_perp_candles(sym["perp"], "1h")
        except (requests.RequestException, ValueError, KeyError):
            perp_ok = False
            perp = spot.rename(columns={"spot_close": "perp_close"})

        merged = (
            spot.merge(perp, on="ts", how="inner")
            .merge(funding, on="ts", how="inner")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        merged["time"] = pd.to_datetime(merged["ts"], unit="ms", utc=True)
        merged.attrs["perp_real"] = perp_ok
        return merged[["ts", "time", "perp_close", "spot_close", "funding_rate"]]
