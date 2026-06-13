"""Market-neutral funding carry backtest.

A carry sign ``s`` on bar t means:

    s = +1  -> short perp + long spot   (harvest positive funding)
    s = -1  -> long perp + short spot   (harvest negative funding)
    s =  0  -> flat

Positions are established on the prior bar (shifted to avoid look-ahead). Per-bar
PnL for a unit carry, with perp leg ``-s`` and spot leg ``+s * hedge_ratio``:

    gross = s * (funding + hedge_ratio * spot_return - perp_return)

The ``hedge_ratio * spot_return - perp_return`` term is the residual basis. When
the perp tracks spot it is ~0, leaving funding as the dominant PnL — that is what
makes this market-neutral rather than directional. Fees are charged on BOTH legs
whenever the carry sign changes.

Known limitation: shorting spot (s = -1) assumes borrow is available and ignores
borrow cost. For spot-margin-constrained venues, cap the signal at s in {0, +1}.
"""
from __future__ import annotations

import pandas as pd

from .engine import BacktestResult
from .metrics import compute_metrics

HOURS_PER_YEAR = 8760


def run_carry_backtest(
    market: pd.DataFrame,
    carry_sign: pd.Series,
    perp_fee_bps: float = 5.0,
    spot_fee_bps: float = 10.0,
    hedge_ratio: float = 1.0,
    periods_per_year: int = HOURS_PER_YEAR,
) -> BacktestResult:
    """Run a two-leg (perp + spot) market-neutral funding carry backtest.

    Args:
        market: frame with ``perp_close``, ``spot_close``, ``funding_rate``.
        carry_sign: target carry sign per bar in {-1, 0, 1}, aligned to ``market``.
        perp_fee_bps: per-side perp transaction cost in basis points.
        spot_fee_bps: per-side spot transaction cost in basis points.
        hedge_ratio: spot notional per unit perp notional (1.0 = fully hedged).
        periods_per_year: bars per year for annualization (8760 for 1h bars).
    """
    market = market.reset_index(drop=True)
    sign = carry_sign.reset_index(drop=True).astype(float)

    perp_return = market["perp_close"].astype(float).pct_change().fillna(0.0)
    spot_return = market["spot_close"].astype(float).pct_change().fillna(0.0)
    funding = market["funding_rate"].astype(float).fillna(0.0)

    held = sign.shift(1).fillna(0.0)  # carry established on the prior bar
    gross = held * (funding + hedge_ratio * spot_return - perp_return)

    turnover = held.diff().abs().fillna(held.abs())
    fees = turnover * ((perp_fee_bps + hedge_ratio * spot_fee_bps) / 1e4)
    net = gross - fees
    equity = (1.0 + net).cumprod()

    metrics = compute_metrics(net, equity, held, periods_per_year)
    return BacktestResult(equity=equity, returns=net, positions=held, metrics=metrics)
