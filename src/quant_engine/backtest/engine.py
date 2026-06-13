"""Vectorized perpetual backtest with funding-aware PnL.

The target position decided on bar t is applied to the return from t to t+1
(positions are shifted by one bar) to avoid look-ahead. Perpetual holders pay
funding when long and funding is positive, so the funding leg is subtracted as
``held * funding_rate``. Transaction costs are charged on turnover.

This is the bootstrap engine — transparent and dependency-light. The documented
upgrade path is vectorbt or QuantConnect LEAN once strategies stabilize.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import PerformanceMetrics, compute_metrics

HOURS_PER_YEAR = 8760


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: PerformanceMetrics


def run_backtest(
    market: pd.DataFrame,
    positions: pd.Series,
    fee_bps: float = 5.0,
    periods_per_year: int = HOURS_PER_YEAR,
    include_funding: bool = True,
) -> BacktestResult:
    """Run a single-instrument vectorized backtest.

    Args:
        market: frame with a ``close`` column and optional ``funding_rate``.
        positions: target position per bar in [-1, 1], aligned to ``market``.
        fee_bps: per-side transaction cost in basis points of traded notional.
        periods_per_year: bars per year for annualization (8760 for 1h bars).
        include_funding: subtract the funding leg when a ``funding_rate`` exists.
    """
    market = market.reset_index(drop=True)
    positions = positions.reset_index(drop=True).astype(float)
    close = market["close"].astype(float)

    price_return = close.pct_change().fillna(0.0)
    held = positions.shift(1).fillna(0.0)  # position established on the prior bar
    gross = held * price_return

    if include_funding and "funding_rate" in market.columns:
        funding = market["funding_rate"].astype(float).fillna(0.0)
        gross = gross - held * funding

    turnover = held.diff().abs().fillna(held.abs())
    fees = turnover * (fee_bps / 1e4)
    net = gross - fees
    equity = (1.0 + net).cumprod()

    metrics = compute_metrics(net, equity, held, periods_per_year)
    return BacktestResult(equity=equity, returns=net, positions=held, metrics=metrics)
