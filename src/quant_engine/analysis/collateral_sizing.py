"""HJB-motivated collateral sizing for spot-perpetual basis trades.

Strategy: hold (1-α)·D in spot, allocate α·D to perpetual margin (short perp).
The short perp earns funding on the spot leg at rate κ̃_h per hour.

Liquidation of the short perp occurs when spot price ratio r = p_t/p_0 exceeds:
    r_liq(α) = 1 / ((1-α) · (1 + θ_F))
where θ_F is the venue maintenance margin fraction (Hyperliquid: 1/(2·L_max)).

Derivation: margin balance = D·[1 − (1−α)·r]; maintenance = θ_F·(1−α)·D·r.
Liquidation when balance ≤ maintenance → r ≥ 1/((1−α)·(1+θ_F)).

Zero-drift GBM first-passage probability:
    Π_liq(α; h) = 2·Φ(−log(r_liq(α)) / (σ_h · √h))
where σ_h is per-hour log-return vol, h is review horizon in hours.

Two control problems (arXiv:2605.05089):
  1. Risk-constrained (recommended): min α  s.t. Π_liq(α;h) ≤ ε
     Solved by bisection (Π_liq is monotone decreasing in α).
  2. Economic optimum: max_α  (1−α)·κ̃_h − LGD·Π_liq(α;h)
     Solved by bisecting the first-order condition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Stdlib-only normal CDF / PDF (no scipy needed)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def theta_F_from_max_leverage(l_max: float) -> float:
    """Maintenance margin fraction from max leverage (Hyperliquid formula: 1/(2·L_max))."""
    return 1.0 / (2.0 * l_max)


def liq_barrier(alpha: float, theta_F: float) -> float:
    """Upper spot-ratio barrier: short perp liquidated if p_t/p_0 ≥ r_liq."""
    denom = (1.0 - alpha) * (1.0 + theta_F)
    if denom <= 0:
        return float("inf")
    return 1.0 / denom


def liq_prob(alpha: float, sigma_h: float, h_hours: float, theta_F: float) -> float:
    """P(short perp liquidated before h hours) via zero-drift GBM first-passage."""
    r_liq = liq_barrier(alpha, theta_F)
    if r_liq <= 1.0:
        return 1.0
    b = math.log(r_liq)                      # log barrier (positive since r_liq > 1)
    vol_horizon = sigma_h * math.sqrt(h_hours)
    if vol_horizon <= 0:
        return 0.0
    z = b / vol_horizon
    return 2.0 * (1.0 - _norm_cdf(z))


def _liq_prob_gradient(alpha: float, sigma_h: float, h_hours: float, theta_F: float) -> float:
    """∂Π_liq/∂α = −2·φ(z) / ((1−α)·σ_h·√h)."""
    r_liq = liq_barrier(alpha, theta_F)
    if r_liq <= 1.0:
        return 0.0
    b = math.log(r_liq)
    vol_horizon = sigma_h * math.sqrt(h_hours)
    if vol_horizon <= 0 or (1.0 - alpha) <= 0:
        return 0.0
    z = b / vol_horizon
    return -2.0 * _norm_pdf(z) / ((1.0 - alpha) * vol_horizon)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def alpha_risk_constrained(
    eps: float,
    sigma_h: float,
    h_hours: float,
    theta_F: float,
    tol: float = 1e-9,
) -> float:
    """Minimum α such that Π_liq(α;h) ≤ ε. Bisection over [0, 1).

    Returns NaN for eps ≤ 0 (zero or negative probability is mathematically
    infeasible since the short perp always has non-zero liquidation risk at
    finite α; note that float64 underflow makes Π_liq = 0 at α→1, so we
    treat eps ≤ 0 as the true infeasibility guard rather than checking the
    numerical value at alpha_max).
    """
    if eps <= 0.0:
        return float("nan")
    alpha_max = 1.0 - tol
    if liq_prob(0.0, sigma_h, h_hours, theta_F) <= eps:
        return 0.0

    lo, hi = 0.0, alpha_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if liq_prob(mid, sigma_h, h_hours, theta_F) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def alpha_economic(
    kappa_h: float,
    lgd: float,
    sigma_h: float,
    h_hours: float,
    theta_F: float,
    tol: float = 1e-9,
) -> float:
    """α* maximising (1−α)·κ̃_h − LGD·Π_liq(α;h).

    FOC: −κ̃_h + LGD·2·φ(z)/((1−α)·σ_h·√h) = 0
    Solved by bisecting g(α) = LGD·|∂Π_liq/∂α| − κ̃_h.

    Search starts from alpha_crit = θ_F/(1+θ_F), the minimum α at which
    r_liq > 1 (below this the gradient is 0 and Π_liq = 1.0 flat — the
    economic objective is dominated by the LGD term and is clearly non-optimal).
    """
    if kappa_h <= 0:
        return 0.0

    # Minimum viable alpha: below this r_liq ≤ 1 → Π_liq = 1.0 (certain liquidation)
    alpha_crit = theta_F / (1.0 + theta_F) + tol
    alpha_max = 1.0 - tol

    if alpha_crit >= alpha_max:
        return alpha_max

    def foc(alpha: float) -> float:
        # g(α) = LGD·|∂Π_liq/∂α| − κ̃_h  (positive → still worth adding margin)
        return lgd * abs(_liq_prob_gradient(alpha, sigma_h, h_hours, theta_F)) - kappa_h

    if foc(alpha_crit) <= 0:
        return alpha_crit   # carry always exceeds LGD cost: use minimum viable margin
    if foc(alpha_max) >= 0:
        return alpha_max    # LGD cost always exceeds carry: use maximum margin

    lo, hi = alpha_crit, alpha_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if foc(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CollateralResult:
    asset: str
    method: str             # 'risk_constrained' | 'economic'
    eps: float              # target max liquidation prob (NaN for economic)
    stress_mult: float      # σ multiplier applied to base vol
    sigma_h_base: float     # base hourly vol (log-return std)
    sigma_h_stressed: float # sigma_h_base * stress_mult
    kappa_h: float          # expected hourly funding carry (fractional rate)
    theta_F: float          # maintenance margin fraction
    h_hours: float          # review horizon in hours
    alpha_star: float       # optimal margin fraction
    pi_liq: float           # Π_liq at alpha_star
    r_liq: float            # spot-ratio liquidation barrier at alpha_star
    carry_net_hourly: float # (1−alpha_star)·kappa_h  (hourly net carry)
    carry_annual_bps: float # carry_net_hourly * 8760 * 10_000

    @property
    def spot_fraction(self) -> float:
        return 1.0 - self.alpha_star

    def one_line(self) -> str:
        tag = f"ε={self.eps:.0%}" if self.method == "risk_constrained" else "econ"
        return (
            f"{self.asset:>5} {self.stress_mult:.1f}× {tag:>6}  "
            f"α*={self.alpha_star:.3f}  Π_liq={self.pi_liq:.4%}  "
            f"carry={self.carry_annual_bps:.1f}bps/yr  r_liq={self.r_liq:.3f}"
        )


# ---------------------------------------------------------------------------
# Batch sizing
# ---------------------------------------------------------------------------

def size_collateral(
    asset: str,
    sigma_h: float,
    theta_F: float,
    kappa_h: float,
    eps_levels: tuple[float, ...] = (0.01, 0.05, 0.10),
    stress_mults: tuple[float, ...] = (1.0, 1.5, 2.0),
    h_hours: float = 24.0,
    lgd: float = 0.5,
) -> list[CollateralResult]:
    """Compute risk-constrained and economic α* for every (stress, ε) combination.

    Args:
        asset: ticker label.
        sigma_h: base hourly log-return vol.
        theta_F: maintenance margin fraction (venue-specific).
        kappa_h: expected hourly funding carry (fractional rate, must be > 0).
        eps_levels: target max liquidation probabilities for risk-constrained variant.
        stress_mults: vol stress multipliers.
        h_hours: review horizon in hours (default 24 = daily).
        lgd: loss given default as fraction of position value (default 0.5).

    Returns:
        Flat list of CollateralResult, one per (stress_mult, method/ε) combination.
    """
    results: list[CollateralResult] = []

    for mult in stress_mults:
        sigma_stressed = sigma_h * mult

        # Risk-constrained variants
        for eps in eps_levels:
            alpha = alpha_risk_constrained(eps, sigma_stressed, h_hours, theta_F)
            if math.isnan(alpha):
                continue
            pi = liq_prob(alpha, sigma_stressed, h_hours, theta_F)
            r = liq_barrier(alpha, theta_F)
            carry_h = (1.0 - alpha) * kappa_h
            results.append(CollateralResult(
                asset=asset,
                method="risk_constrained",
                eps=eps,
                stress_mult=mult,
                sigma_h_base=sigma_h,
                sigma_h_stressed=sigma_stressed,
                kappa_h=kappa_h,
                theta_F=theta_F,
                h_hours=h_hours,
                alpha_star=alpha,
                pi_liq=pi,
                r_liq=r,
                carry_net_hourly=carry_h,
                carry_annual_bps=carry_h * 8760.0 * 10_000.0,
            ))

        # Economic optimum
        if kappa_h > 0:
            alpha = alpha_economic(kappa_h, lgd, sigma_stressed, h_hours, theta_F)
            if not math.isnan(alpha):
                pi = liq_prob(alpha, sigma_stressed, h_hours, theta_F)
                r = liq_barrier(alpha, theta_F)
                carry_h = (1.0 - alpha) * kappa_h
                results.append(CollateralResult(
                    asset=asset,
                    method="economic",
                    eps=float("nan"),
                    stress_mult=mult,
                    sigma_h_base=sigma_h,
                    sigma_h_stressed=sigma_stressed,
                    kappa_h=kappa_h,
                    theta_F=theta_F,
                    h_hours=h_hours,
                    alpha_star=alpha,
                    pi_liq=pi,
                    r_liq=r,
                    carry_net_hourly=carry_h,
                    carry_annual_bps=carry_h * 8760.0 * 10_000.0,
                ))

    return results


def results_to_table(results: list[CollateralResult]) -> str:
    """Format results as a grid: stress rows × ε/method columns."""
    from collections import defaultdict
    import math as _math

    eps_cols = sorted({r.eps for r in results if r.method == "risk_constrained"})
    has_econ = any(r.method == "economic" for r in results)
    assets = list(dict.fromkeys(r.asset for r in results))
    mults = sorted(dict.fromkeys(r.stress_mult for r in results))

    col_labels = [f"ε={e:.0%}" for e in eps_cols] + (["econ"] if has_econ else [])
    header = f"{'Asset':>6} {'Stress':>7}  " + "  ".join(f"{c:>14}" for c in col_labels)
    sep = "-" * len(header)
    rows = [header, sep]

    for asset in assets:
        for mult in mults:
            sub = {r.eps: r for r in results if r.asset == asset and r.stress_mult == mult}
            econ = next((r for r in results
                         if r.asset == asset and r.stress_mult == mult
                         and r.method == "economic"), None)
            cells = []
            for eps in eps_cols:
                res = sub.get(eps)
                if res is None:
                    cells.append(f"{'N/A':>14}")
                else:
                    cells.append(f"α={res.alpha_star:.3f}|{res.carry_annual_bps:>5.0f}bp")
            if has_econ:
                if econ is None:
                    cells.append(f"{'N/A':>14}")
                else:
                    cells.append(f"α={econ.alpha_star:.3f}|{econ.carry_annual_bps:>5.0f}bp")
            rows.append(f"{asset:>6} {mult:.1f}×      " + "  ".join(cells))
        rows.append("")

    rows.append("  α* = optimal margin fraction | carry = annualised net (bps/yr)")
    return "\n".join(rows)
