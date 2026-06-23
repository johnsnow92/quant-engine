"""Ornstein-Uhlenbeck calibration for Hyperliquid funding rates.

Fits kappa (mean-reversion speed), theta_bar (long-run mean), and sigma_f
(diffusion volatility) to hourly funding observations via the closed-form
OLS/MLE estimator for Gaussian OU.  No external optimiser needed.

The discrete-time OU transition (Delta = 1 hour):

    F_{t+1} | F_t ~ N(theta + (F_t - theta)*exp(-kappa), sigma^2*(1 - exp(-2*kappa)) / (2*kappa))

is an AR(1):

    F_{t+1} = a + b*F_t + eps_t,   eps_t ~ N(0, s^2)

OLS on this regression is the exact MLE for Gaussian OU.  Recovery:

    b  = exp(-kappa)        ->  kappa    = -log(b)
    a  = theta*(1 - b)      ->  theta    = a / (1 - b)
    s^2 = sigma^2*(1-b^2)   ->  sigma_f  = sqrt(2*kappa*s^2 / (1 - b^2))
           / (2*kappa)

Jump proxy: fraction of |residuals| > 3*s per hour.

Stationary distribution: N(theta, sigma_f^2 / (2*kappa))
    -> useful for HJB grid bounds: theta +/- n_sigma * sigma_stationary

Reference: paper arXiv:2605.06405, Section 6.2 (Hyperliquid ETH/BTC/SOL).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    """Calibrated OU parameters for one asset/window."""

    asset: str
    window_days: int
    n_obs: int

    # core OU params (fractional funding rate, per-hour units)
    kappa: float       # mean-reversion speed (h^-1)
    theta: float       # long-run mean (fractional rate)
    sigma_f: float     # diffusion vol (fractional rate / sqrt(h))

    # derived
    half_life_hours: float    # log(2) / kappa
    sigma_stationary: float   # sigma_f / sqrt(2*kappa)  -- stationary std dev

    # diagnostics
    ar1_b: float              # AR(1) slope (= exp(-kappa))
    residual_std: float       # s from OLS residuals
    jump_prob_per_hour: float # fraction of |eps| > 3*s

    # HJB grid suggestion: theta +/- grid_n_sigma * sigma_stationary
    grid_n_sigma: float = 4.0
    f_grid_min: float = field(init=False)
    f_grid_max: float = field(init=False)

    def __post_init__(self) -> None:
        self.f_grid_min = self.theta - self.grid_n_sigma * self.sigma_stationary
        self.f_grid_max = self.theta + self.grid_n_sigma * self.sigma_stationary

    def summary(self) -> str:
        lines = [
            f"Asset={self.asset}  window={self.window_days}d  n={self.n_obs}",
            f"  kappa      = {self.kappa:.6f} h⁻¹   (half-life = {self.half_life_hours:.2f} h)",
            f"  theta      = {self.theta:.6e}  (annualised = {self.theta * 8760:.2%})",
            f"  sigma_f    = {self.sigma_f:.6e}",
            f"  sigma_stat = {self.sigma_stationary:.6e}",
            f"  jump proxy = {self.jump_prob_per_hour:.2%}/h",
            f"  HJB grid   [{self.f_grid_min:.4e}, {self.f_grid_max:.4e}]",
        ]
        return "\n".join(lines)


def _fit_ou(rates: np.ndarray) -> tuple[float, float, float, float, float]:
    """OLS AR(1) -> recover (kappa, theta, sigma_f, residual_std, b).

    rates: 1-D array of hourly fractional funding rates, Δ = 1 h.
    Returns (kappa, theta, sigma_f, s, b).
    """
    y = rates[1:]
    x = rates[:-1]
    n = len(y)

    # OLS: y = a + b*x + eps
    x_bar = x.mean()
    y_bar = y.mean()
    sxy = ((x - x_bar) * (y - y_bar)).sum()
    sxx = ((x - x_bar) ** 2).sum()

    b = sxy / sxx
    a = y_bar - b * x_bar

    residuals = y - (a + b * x)
    s2 = residuals.var(ddof=2)          # unbiased
    s = math.sqrt(max(s2, 1e-30))

    # guard against b >= 1 (non-stationary) or b <= 0 (over-mean-reverting)
    b = float(np.clip(b, 1e-6, 1 - 1e-9))

    kappa = -math.log(b)                # per hour (Δ = 1 h)
    theta = a / (1.0 - b)

    # sigma_f: s^2 = sigma_f^2 * (1 - b^2) / (2*kappa)
    # => sigma_f = sqrt(2*kappa*s^2 / (1 - b^2))
    sigma_f = math.sqrt(2.0 * kappa * s2 / (1.0 - b ** 2))

    return kappa, theta, sigma_f, s, b, residuals


def calibrate(
    funding_df: pd.DataFrame,
    asset: str,
    window_days: int = 180,
    rate_col: str = "funding_rate",
    ts_col: str = "ts",
) -> OUParams:
    """Calibrate OU params from a funding history DataFrame.

    Args:
        funding_df: DataFrame with columns [ts_col (ms), rate_col].
        asset: ticker label (e.g. 'ETH').
        window_days: lookback window in calendar days.
        rate_col: column name for the fractional hourly funding rate.
        ts_col: column name for unix-ms timestamps.

    Returns:
        OUParams with calibrated kappa, theta, sigma_f, and diagnostics.
    """
    df = funding_df.copy().sort_values(ts_col).reset_index(drop=True)
    cutoff_ms = df[ts_col].max() - window_days * 86_400_000
    df = df[df[ts_col] >= cutoff_ms]

    rates = df[rate_col].dropna().to_numpy(dtype=float)
    if len(rates) < 10:
        raise ValueError(f"Too few observations ({len(rates)}) for {asset} window={window_days}d")

    kappa, theta, sigma_f, s, b, residuals = _fit_ou(rates)

    half_life = math.log(2.0) / kappa
    sigma_stat = sigma_f / math.sqrt(2.0 * kappa)
    jump_prob = float((np.abs(residuals) > 3.0 * s).mean())

    return OUParams(
        asset=asset,
        window_days=window_days,
        n_obs=len(rates),
        kappa=kappa,
        theta=theta,
        sigma_f=sigma_f,
        half_life_hours=half_life,
        sigma_stationary=sigma_stat,
        ar1_b=b,
        residual_std=s,
        jump_prob_per_hour=jump_prob,
    )


def calibrate_multi_window(
    funding_df: pd.DataFrame,
    asset: str,
    windows: tuple[int, ...] = (90, 180, 360),
    rate_col: str = "funding_rate",
    ts_col: str = "ts",
) -> list[OUParams]:
    """Run calibrate() for each window; return list ordered by window size."""
    results = []
    for w in windows:
        try:
            results.append(calibrate(funding_df, asset, w, rate_col, ts_col))
        except ValueError as exc:
            print(f"  [warn] {exc}")
    return results


def params_to_dict(p: OUParams) -> dict:
    """Serialise OUParams to a plain dict (JSON-safe)."""
    return {
        "asset": p.asset,
        "window_days": p.window_days,
        "n_obs": p.n_obs,
        "kappa": p.kappa,
        "theta": p.theta,
        "sigma_f": p.sigma_f,
        "half_life_hours": p.half_life_hours,
        "sigma_stationary": p.sigma_stationary,
        "ar1_b": p.ar1_b,
        "residual_std": p.residual_std,
        "jump_prob_per_hour": p.jump_prob_per_hour,
        "f_grid_min": p.f_grid_min,
        "f_grid_max": p.f_grid_max,
    }
