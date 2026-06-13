"""Performance metrics for a backtest equity curve."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    exposure: float


def price_beta(returns: pd.Series, price_returns: pd.Series) -> float:
    """Sensitivity of strategy returns to the underlying price return.

    ~0 means market-neutral; a large magnitude means the strategy still carries
    directional price exposure. Used to confirm a hedge is actually neutralizing
    the price leg.
    """
    a = returns.to_numpy(dtype=float)
    b = price_returns.to_numpy(dtype=float)
    var = float(b.var())
    if var == 0.0 or len(a) != len(b):
        return 0.0
    return float(np.cov(a, b, ddof=0)[0, 1] / var)


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline of the equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    positions: pd.Series,
    periods_per_year: int,
) -> PerformanceMetrics:
    returns = returns.fillna(0.0)
    n = len(returns)
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    ann_return = float((1.0 + returns.mean()) ** periods_per_year - 1.0) if n else 0.0
    vol = float(returns.std(ddof=0))
    ann_vol = vol * np.sqrt(periods_per_year)
    sharpe = float(returns.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0
    trades = int((positions.diff().fillna(positions).abs() > 1e-9).sum())
    exposure = float((positions.abs() > 1e-9).mean()) if len(positions) else 0.0
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown(equity),
        num_trades=trades,
        exposure=exposure,
    )
