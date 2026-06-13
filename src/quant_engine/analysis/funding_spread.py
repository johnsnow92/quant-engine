"""Cross-venue funding-rate comparison and carry.

Different venues fund on different schedules (e.g. Crypto.com hourly, Binance
8-hourly), so raw funding rates are not comparable. Everything here normalizes to
a per-hour rate first, then compares.

Cross-venue funding carry: hold a BTC perp on the low-funding venue and the
opposite BTC perp on the high-funding venue. Both legs track BTC, so price
exposure roughly cancels, and you collect the funding spread.

Approximation: per-hour normalized funding is treated as continuously accrued.
Funding is actually charged at discrete settlement times, and cross-venue basis
and transfer/borrow frictions are ignored. This sizes the opportunity; it is not
a fill-accurate model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..backtest.metrics import compute_metrics

HOURS_PER_YEAR = 8760


def normalize_per_hour(
    funding: pd.DataFrame,
    rate_col: str = "funding_rate",
    interval_col: str = "interval_hours",
) -> pd.DataFrame:
    """Add a ``funding_per_hour`` column = rate / settlement interval (hours)."""
    out = funding.copy()
    out["funding_per_hour"] = out[rate_col] / out[interval_col]
    return out


def annualized(per_hour_rate: float) -> float:
    return per_hour_rate * HOURS_PER_YEAR


def build_spread(venue_a: pd.DataFrame, venue_b: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two venues on timestamp; spread = b_per_hour - a_per_hour."""
    a = venue_a[["ts", "funding_per_hour"]].rename(columns={"funding_per_hour": "a_per_hour"})
    b = venue_b[["ts", "funding_per_hour"]].rename(columns={"funding_per_hour": "b_per_hour"})
    merged = a.merge(b, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    merged["spread_per_hour"] = merged["b_per_hour"] - merged["a_per_hour"]
    return merged


def cross_venue_carry(
    spread: pd.DataFrame,
    rebalance_fee_bps: float = 2.0,
    periods_per_year: int = HOURS_PER_YEAR,
) -> BacktestResult:
    """Backtest collecting the funding spread, positioned to always harvest it.

    Each bar, take the side of the spread (short the higher-funding venue, long the
    lower) decided on the prior bar. Gross per bar = |spread|; fees are charged on
    BOTH perp legs whenever the side flips.
    """
    s = spread["spread_per_hour"].reset_index(drop=True).astype(float)
    held = pd.Series(np.sign(s)).shift(1).fillna(0.0)
    gross = held * s
    turnover = held.diff().abs().fillna(held.abs())
    fees = turnover * (2 * rebalance_fee_bps / 1e4)  # two perp legs
    net = gross - fees
    equity = (1.0 + net).cumprod()
    metrics = compute_metrics(net, equity, held, periods_per_year)
    return BacktestResult(equity=equity, returns=net, positions=held, metrics=metrics)
