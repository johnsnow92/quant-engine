"""Funding-regime detector for the quant-engine wake trigger.

A 'regime ON' means the rolling 24h annualized funding rate on at least one
venue clears the hurdle rate, or the cross-venue spread clears a separate
threshold — making the carry strategy worth running.

Regime thresholds are intentionally conservative relative to the 4.70% LOC
floor: both the VOO hurdle (~14%) and a smaller carry-spread floor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ..analysis.funding_spread import annualized, build_spread, normalize_per_hour
from ..data.coindesk import CoinDeskClient
from ..data.cryptocom import CryptoComClient

log = logging.getLogger(__name__)

# Default 15 % annualized — above the VOO portfolio benchmark.
DEFAULT_SINGLE_VENUE_THRESHOLD = 0.15
# Cross-venue spread only needs to clear transaction costs by a margin.
DEFAULT_SPREAD_THRESHOLD = 0.05
LOOKBACK_BARS = 24  # 24-hour rolling window (1h bars)


@dataclass
class RegimeState:
    is_on: bool
    btc_cc_ann: float       # Crypto.com BTC annualized (last 24h mean)
    btc_bn_ann: float       # Binance BTC annualized (last 24h mean)
    btc_spread_ann: float   # Cross-venue spread annualized
    eth_cc_ann: float
    triggered_by: list[str]  # which condition(s) fired
    checked_at: datetime

    def summary(self) -> str:
        state = "ON" if self.is_on else "OFF"
        lines = [
            f"Funding regime: {state}  ({self.checked_at.strftime('%Y-%m-%d %H:%M UTC')})",
            f"  BTC Crypto.com {self.btc_cc_ann:+.1%} ann",
            f"  BTC Binance    {self.btc_bn_ann:+.1%} ann",
            f"  BTC spread     {self.btc_spread_ann:+.1%} ann",
            f"  ETH Crypto.com {self.eth_cc_ann:+.1%} ann",
        ]
        if self.triggered_by:
            lines.append(f"  Triggered by: {', '.join(self.triggered_by)}")
        return "\n".join(lines)


def _rolling_mean_ann(df: pd.DataFrame, col: str = "funding_per_hour", n: int = LOOKBACK_BARS) -> float:
    tail = df[col].dropna().tail(n)
    if tail.empty:
        return 0.0
    return annualized(tail.mean())


def check_regime(
    single_venue_threshold: float = DEFAULT_SINGLE_VENUE_THRESHOLD,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
    lookback: int = LOOKBACK_BARS,
    cc_interval_hours: float = 1.0,
    cc_btc_perp: str = "BTCUSD-PERP",
    cc_eth_perp: str = "ETHUSD-PERP",
    bn_btc_instrument: str = "BTC-USDT-VANILLA-PERPETUAL",
    bn_market: str = "binance",
    count: int = 300,
) -> RegimeState:
    """Fetch live funding from Crypto.com (required) + Binance/CoinDesk (optional).

    CoinDesk requires COINDESK_API_KEY for funding data. If unavailable, the
    cross-venue spread check is skipped and single-venue Crypto.com data is used.
    """
    cc = CryptoComClient()

    log.debug("Fetching Crypto.com BTC funding...")
    cc_btc = cc.fetch_funding(cc_btc_perp, count=count)
    cc_btc["interval_hours"] = cc_interval_hours
    cc_btc = normalize_per_hour(cc_btc)

    log.debug("Fetching Crypto.com ETH funding...")
    cc_eth = cc.fetch_funding(cc_eth_perp, count=count)
    cc_eth["interval_hours"] = cc_interval_hours
    cc_eth = normalize_per_hour(cc_eth)

    btc_cc_ann = _rolling_mean_ann(cc_btc, n=lookback)
    eth_cc_ann = _rolling_mean_ann(cc_eth, n=lookback)

    # Binance cross-venue spread — requires COINDESK_API_KEY.
    btc_bn_ann = 0.0
    btc_spread_ann = 0.0
    try:
        cd = CoinDeskClient()
        log.debug("Fetching Binance BTC funding via CoinDesk...")
        bn_btc = cd.fetch_funding(bn_market, bn_btc_instrument, limit=count)
        bn_btc = normalize_per_hour(bn_btc)
        btc_bn_ann = _rolling_mean_ann(bn_btc, n=lookback)
        spread = build_spread(cc_btc, bn_btc)
        btc_spread_ann = _rolling_mean_ann(spread, col="spread_per_hour", n=lookback)
    except Exception as exc:
        log.info("CoinDesk/Binance unavailable (set COINDESK_API_KEY to enable): %s", exc)

    triggered: list[str] = []
    if abs(btc_cc_ann) >= single_venue_threshold:
        triggered.append(f"BTC Crypto.com {btc_cc_ann:+.1%}")
    if btc_bn_ann and abs(btc_bn_ann) >= single_venue_threshold:
        triggered.append(f"BTC Binance {btc_bn_ann:+.1%}")
    if abs(eth_cc_ann) >= single_venue_threshold:
        triggered.append(f"ETH Crypto.com {eth_cc_ann:+.1%}")
    if btc_spread_ann and abs(btc_spread_ann) >= spread_threshold:
        triggered.append(f"BTC spread {btc_spread_ann:+.1%}")

    return RegimeState(
        is_on=bool(triggered),
        btc_cc_ann=btc_cc_ann,
        btc_bn_ann=btc_bn_ann,
        btc_spread_ann=btc_spread_ann,
        eth_cc_ann=eth_cc_ann,
        triggered_by=triggered,
        checked_at=datetime.now(timezone.utc),
    )
