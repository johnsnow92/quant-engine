"""Target-portfolio sizer for the live spot-perp carry strategy.

Translates the backtest-locked strategy (risk-constrained collateral sizing at
ε=1% with a 2× vol-stress multiplier — the config that stayed zero-liquidation
across the full funding cycle) into a concrete target for the current account
equity: how much spot to hold long and how much perp to short so the book is
delta-neutral and the short perp's liquidation probability over the review
horizon stays ≤ ε.

Pure, offline logic — no network, no orders. It produces the target that the
order planner reconciles against the live position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from quant_engine.analysis.collateral_sizing import (
    alpha_risk_constrained,
    liq_barrier,
    liq_prob,
)


@dataclass
class ExecutionConfig:
    """Production sizing parameters — the backtest-locked winner (ε=1%, 2× stress)."""
    eps: float = 0.01            # max liquidation probability per review horizon
    stress_mult: float = 2.0     # vol stress multiplier on realized σ_h
    theta_F: float = 0.01        # venue maintenance margin fraction (Hyperliquid 50×)
    h_hours: float = 24.0        # review horizon for the liquidation-probability constraint
    vol_lookback_hours: int = 720  # rolling window for σ_h (30 days of hourly bars)


@dataclass
class TargetPortfolio:
    asset: str
    equity: float
    alpha: float             # margin fraction α* (perp margin / equity)
    sigma_h: float           # realized hourly log-return vol used for sizing
    spot_price: float
    perp_price: float
    spot_notional: float     # (1-α)·equity, held long in spot
    perp_notional: float     # matched short notional (delta-neutral)
    perp_margin: float       # α·equity posted as perp margin
    spot_qty: float          # asset units, long (+)
    perp_qty: float          # asset units, short (−), equal magnitude to spot_qty
    r_liq: float             # spot-ratio liquidation barrier at α*
    barrier_move_pct: float  # r_liq − 1 (the upward move that would liquidate)
    pi_liq: float            # liquidation probability at α* over h_hours


def compute_sigma_h(prices) -> float:
    """Hourly log-return std from a price series."""
    arr = np.asarray(prices, dtype=float)
    if arr.size < 2:
        raise ValueError("need >= 2 prices to estimate volatility")
    log_rets = np.diff(np.log(arr))
    return float(log_rets.std(ddof=1))


def compute_target(
    asset: str,
    equity: float,
    price_history,
    spot_price: float,
    perp_price: float,
    config: ExecutionConfig | None = None,
) -> TargetPortfolio:
    """Compute the delta-neutral target portfolio for the current equity.

    Args:
        asset: ticker (e.g. 'ETH').
        equity: account equity in quote currency (USD).
        price_history: recent hourly prices for the vol estimate (most recent last);
            the last ``vol_lookback_hours+1`` points are used.
        spot_price: current spot price (for the long leg).
        perp_price: current perp mark (for the short leg).
        config: ExecutionConfig (defaults to the locked ε=1% / 2× winner).

    Returns:
        TargetPortfolio with the target quantities and risk diagnostics.
    """
    config = config or ExecutionConfig()
    if equity <= 0:
        raise ValueError("equity must be positive")
    if spot_price <= 0 or perp_price <= 0:
        raise ValueError("prices must be positive")

    window = np.asarray(price_history, dtype=float)[-(config.vol_lookback_hours + 1):]
    sigma_h = compute_sigma_h(window)
    sigma_stressed = sigma_h * config.stress_mult

    alpha = alpha_risk_constrained(config.eps, sigma_stressed, config.h_hours, config.theta_F)
    if math.isnan(alpha):
        raise ValueError(f"infeasible eps={config.eps} for sizing")

    spot_notional = (1.0 - alpha) * equity
    spot_qty = spot_notional / spot_price
    perp_qty = -spot_qty                       # match quantity → delta-neutral in asset units
    perp_notional = spot_qty * perp_price
    perp_margin = alpha * equity

    return TargetPortfolio(
        asset=asset,
        equity=equity,
        alpha=alpha,
        sigma_h=sigma_h,
        spot_price=spot_price,
        perp_price=perp_price,
        spot_notional=spot_notional,
        perp_notional=perp_notional,
        perp_margin=perp_margin,
        spot_qty=spot_qty,
        perp_qty=perp_qty,
        r_liq=liq_barrier(alpha, config.theta_F),
        barrier_move_pct=liq_barrier(alpha, config.theta_F) - 1.0,
        pi_liq=liq_prob(alpha, sigma_stressed, config.h_hours, config.theta_F),
    )


# Bitnomial US-perp contract sizes (asset units per contract). US perps trade in
# WHOLE contracts only — fractional contracts are rejected at order entry — so the
# perp leg must be quantized and the spot leg matched to the executable quantity.
BITNOMIAL_CONTRACT_SIZES = {"BTC": 0.01, "ETH": 0.5, "SOL": 5.0}


def quantize_to_contracts(target: TargetPortfolio, contract_size: float) -> TargetPortfolio:
    """Snap a delta-neutral target to whole perp contracts, matching the spot leg.

    The continuous ``target.perp_qty`` is rounded to the nearest whole-contract
    quantity (n = round(|perp_qty| / contract_size)); the long spot leg is then
    matched to that exact quantity so the book stays delta-neutral at the
    executable granularity. The leftover between the ideal (1-α) spot notional and
    the contract-snapped notional is the unavoidable quantization residual — small
    at reasonable equity, coarse when one contract is a large fraction of equity.

    Returns a copy with spot_qty/perp_qty/notionals snapped (α, margin, and risk
    diagnostics are unchanged — quantization moves quantity, not the risk frame).
    """
    if contract_size <= 0:
        raise ValueError("contract_size must be positive")
    n_contracts = round(abs(target.perp_qty) / contract_size)
    qty = n_contracts * contract_size
    return replace(
        target,
        spot_qty=qty,
        perp_qty=-qty,                              # short, matched → delta-neutral
        spot_notional=qty * target.spot_price,
        perp_notional=qty * target.perp_price,
    )
