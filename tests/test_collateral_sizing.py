"""Tests for HJB collateral sizing (analysis/collateral_sizing.py).

Three layers:
  1. Math unit tests — liq_barrier, liq_prob, norm_cdf/pdf identities.
  2. Optimizer tests — bisection correctness for risk-constrained & economic.
  3. Regression locks — load data/collateral_sizing.json and assert key bands.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from quant_engine.analysis.collateral_sizing import (
    _norm_cdf,
    _norm_pdf,
    alpha_economic,
    alpha_risk_constrained,
    liq_barrier,
    liq_prob,
    size_collateral,
    theta_F_from_max_leverage,
    CollateralResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZING_JSON = REPO_ROOT / "data" / "collateral_sizing.json"

# Hyperliquid-style params for all tests
THETA_F_50 = theta_F_from_max_leverage(50.0)   # 0.01
SIGMA_H = 0.01     # 1% hourly — typical BTC/ETH
H = 24.0           # 1-day horizon
KAPPA_H = 5e-6     # realistic hourly funding carry


# ---------------------------------------------------------------------------
# 1. Math unit tests
# ---------------------------------------------------------------------------

class TestNormFunctions:
    def test_cdf_at_zero(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)

    def test_cdf_large_positive(self):
        assert _norm_cdf(10.0) == pytest.approx(1.0, abs=1e-10)

    def test_cdf_large_negative(self):
        assert _norm_cdf(-10.0) == pytest.approx(0.0, abs=1e-10)

    def test_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert _norm_pdf(0.0) == pytest.approx(expected, rel=1e-10)

    def test_pdf_nonnegative(self):
        for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert _norm_pdf(x) >= 0

    def test_cdf_symmetry(self):
        for x in (0.5, 1.0, 2.0):
            assert _norm_cdf(-x) + _norm_cdf(x) == pytest.approx(1.0, abs=1e-12)


class TestLiqBarrier:
    def test_full_margin_infinite_barrier(self):
        assert liq_barrier(1.0 - 1e-12, THETA_F_50) > 1e10

    def test_zero_margin_barrier_below_one(self):
        # α=0 → r_liq = 1/(1+θ_F) < 1 → liquidated immediately
        r = liq_barrier(0.0, THETA_F_50)
        assert r < 1.0

    def test_barrier_increases_in_alpha(self):
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        barriers = [liq_barrier(a, THETA_F_50) for a in alphas]
        assert all(barriers[i] < barriers[i + 1] for i in range(len(barriers) - 1))

    def test_barrier_formula(self):
        alpha, theta_F = 0.4, 0.01
        expected = 1.0 / (0.6 * 1.01)
        assert liq_barrier(alpha, theta_F) == pytest.approx(expected, rel=1e-12)

    def test_theta_F_from_max_leverage_50(self):
        assert theta_F_from_max_leverage(50.0) == pytest.approx(0.01, rel=1e-10)

    def test_theta_F_from_max_leverage_20(self):
        assert theta_F_from_max_leverage(20.0) == pytest.approx(0.025, rel=1e-10)


class TestLiqProb:
    def test_zero_alpha_high_prob(self):
        # With zero margin, barrier is below 1 → certain liquidation
        assert liq_prob(0.0, SIGMA_H, H, THETA_F_50) == pytest.approx(1.0, abs=1e-12)

    def test_prob_decreasing_in_alpha(self):
        probs = [liq_prob(a, SIGMA_H, H, THETA_F_50) for a in [0.1, 0.3, 0.5, 0.7, 0.9]]
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_prob_in_unit_interval(self):
        for alpha in [0.2, 0.5, 0.8]:
            p = liq_prob(alpha, SIGMA_H, H, THETA_F_50)
            assert 0.0 <= p <= 1.0

    def test_prob_zero_vol(self):
        # With zero vol, no diffusion → zero liquidation prob (barrier never hit)
        assert liq_prob(0.5, 0.0, H, THETA_F_50) == 0.0

    def test_known_value(self):
        # Manual: r_liq(0.5) = 1/(0.5*1.01) ≈ 1.9802; b = log(1.9802) ≈ 0.6832
        # vol_h = 0.01*sqrt(24) ≈ 0.04899; z = 0.6832/0.04899 ≈ 13.95
        # Π_liq = 2*(1-Φ(13.95)) ≈ 0 (extremely safe)
        p = liq_prob(0.5, SIGMA_H, H, THETA_F_50)
        assert p < 1e-30


# ---------------------------------------------------------------------------
# 2. Optimizer tests
# ---------------------------------------------------------------------------

class TestAlphaRiskConstrained:
    def test_liq_prob_at_alpha_star_satisfies_eps(self):
        for eps in (0.01, 0.05, 0.10):
            alpha = alpha_risk_constrained(eps, SIGMA_H, H, THETA_F_50)
            if not math.isnan(alpha):
                assert liq_prob(alpha, SIGMA_H, H, THETA_F_50) <= eps + 1e-9

    def test_alpha_increases_with_tighter_eps(self):
        # Tighter ε → more margin required
        a1 = alpha_risk_constrained(0.10, SIGMA_H, H, THETA_F_50)
        a5 = alpha_risk_constrained(0.01, SIGMA_H, H, THETA_F_50)
        assert a5 >= a1

    def test_alpha_increases_with_higher_vol(self):
        a_base = alpha_risk_constrained(0.05, SIGMA_H, H, THETA_F_50)
        a_stress = alpha_risk_constrained(0.05, SIGMA_H * 2.0, H, THETA_F_50)
        assert a_stress >= a_base

    def test_alpha_in_unit_interval(self):
        alpha = alpha_risk_constrained(0.05, SIGMA_H, H, THETA_F_50)
        if not math.isnan(alpha):
            assert 0.0 <= alpha <= 1.0

    def test_returns_nan_for_zero_eps(self):
        # eps ≤ 0 is physically infeasible (probability can't be exactly 0)
        assert math.isnan(alpha_risk_constrained(0.0, SIGMA_H, H, THETA_F_50))
        assert math.isnan(alpha_risk_constrained(-0.01, SIGMA_H, H, THETA_F_50))

    def test_returns_zero_for_easy_eps(self):
        # ε=1.0 (100% tolerated) → α*=0 since even α=0 satisfies
        alpha = alpha_risk_constrained(1.0, SIGMA_H, H, THETA_F_50)
        assert alpha == pytest.approx(0.0, abs=1e-6)


class TestAlphaEconomic:
    def test_alpha_in_unit_interval(self):
        alpha = alpha_economic(KAPPA_H, 0.5, SIGMA_H, H, THETA_F_50)
        assert 0.0 <= alpha <= 1.0

    def test_zero_carry_returns_zero(self):
        assert alpha_economic(0.0, 0.5, SIGMA_H, H, THETA_F_50) == 0.0

    def test_negative_carry_returns_zero(self):
        assert alpha_economic(-1e-6, 0.5, SIGMA_H, H, THETA_F_50) == 0.0

    def test_higher_vol_requires_more_margin(self):
        a_base = alpha_economic(KAPPA_H, 0.5, SIGMA_H, H, THETA_F_50)
        a_stress = alpha_economic(KAPPA_H, 0.5, SIGMA_H * 2.0, H, THETA_F_50)
        assert a_stress >= a_base

    def test_higher_lgd_requires_more_margin(self):
        # Higher LGD → cost of liquidation higher → want lower Π_liq → more α
        a_low_lgd = alpha_economic(KAPPA_H, 0.1, SIGMA_H, H, THETA_F_50)
        a_high_lgd = alpha_economic(KAPPA_H, 0.9, SIGMA_H, H, THETA_F_50)
        assert a_high_lgd >= a_low_lgd

    def test_economic_more_conservative_than_loose_rc(self):
        # With large LGD (0.5) and tiny hourly carry (5e-6), the expected liquidation
        # loss dominates → economic α* is MORE conservative than the loose 10% ε cap.
        a_econ = alpha_economic(KAPPA_H, 0.5, SIGMA_H, H, THETA_F_50)
        a_rc = alpha_risk_constrained(0.10, SIGMA_H, H, THETA_F_50)
        assert a_econ >= a_rc


class TestSizeCollateral:
    def test_returns_list_of_results(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H)
        assert len(results) > 0
        assert all(isinstance(r, CollateralResult) for r in results)

    def test_one_result_per_stress_eps_combo(self):
        eps = (0.01, 0.05)
        mults = (1.0, 1.5)
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H,
                                  eps_levels=eps, stress_mults=mults)
        rc = [r for r in results if r.method == "risk_constrained"]
        econ = [r for r in results if r.method == "economic"]
        assert len(rc) == len(eps) * len(mults)
        assert len(econ) == len(mults)

    def test_carry_net_consistent(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H)
        for r in results:
            expected_carry = (1.0 - r.alpha_star) * r.kappa_h
            assert r.carry_net_hourly == pytest.approx(expected_carry, rel=1e-9)

    def test_annualised_carry_consistent(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H)
        for r in results:
            expected = r.carry_net_hourly * 8760.0 * 10_000.0
            assert r.carry_annual_bps == pytest.approx(expected, rel=1e-9)

    def test_higher_stress_higher_alpha(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H,
                                  eps_levels=(0.05,), stress_mults=(1.0, 2.0))
        rc = {r.stress_mult: r for r in results if r.method == "risk_constrained" and r.eps == 0.05}
        assert rc[2.0].alpha_star >= rc[1.0].alpha_star

    def test_zero_carry_no_economic_results(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, 0.0)
        econ = [r for r in results if r.method == "economic"]
        assert len(econ) == 0

    def test_r_liq_consistent(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H)
        for r in results:
            expected = liq_barrier(r.alpha_star, r.theta_F)
            assert r.r_liq == pytest.approx(expected, rel=1e-9)

    def test_spot_fraction_plus_alpha_is_one(self):
        results = size_collateral("ETH", SIGMA_H, THETA_F_50, KAPPA_H)
        for r in results:
            assert r.spot_fraction + r.alpha_star == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Regression locks — live sizing results (data/collateral_sizing.json)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sizing_json() -> dict:
    if not SIZING_JSON.exists():
        pytest.skip(f"collateral_sizing.json not found — run compute_collateral_sizing.py first")
    with open(SIZING_JSON) as f:
        return json.load(f)


def _find_result(data: dict, asset: str, method: str, stress_mult: float, eps=None) -> dict:
    for r in data["results"]:
        if r["asset"] != asset or r["method"] != method or r["stress_mult"] != stress_mult:
            continue
        if method == "risk_constrained" and r["eps"] != eps:
            continue
        return r
    pytest.skip(f"No result for {asset}/{method}/stress={stress_mult}/eps={eps}")


class TestRegressionLocks:
    """Assert that live sizing results stay within expected bands.

    The bands are intentionally wide — they catch code regressions and
    gross parameter shifts without requiring re-generation every day.
    """

    # --- Structural sanity ---

    def test_results_list_nonempty(self, sizing_json):
        assert len(sizing_json["results"]) > 0

    def test_all_assets_present(self, sizing_json):
        found = {r["asset"] for r in sizing_json["results"]}
        assert {"ETH", "BTC"}.issubset(found)

    def test_alpha_always_in_unit_interval(self, sizing_json):
        for r in sizing_json["results"]:
            assert 0.0 <= r["alpha_star"] <= 1.0, (
                f"alpha_star={r['alpha_star']} out of [0,1] for "
                f"{r['asset']}/{r['method']}/stress={r['stress_mult']}"
            )

    def test_pi_liq_in_unit_interval(self, sizing_json):
        for r in sizing_json["results"]:
            assert 0.0 <= r["pi_liq"] <= 1.0

    def test_r_liq_above_one(self, sizing_json):
        for r in sizing_json["results"]:
            assert r["r_liq"] > 1.0, f"r_liq={r['r_liq']} ≤ 1 for {r['asset']}"

    # --- Risk-constrained: eps constraint satisfied ---

    def test_eth_rc_1pct_eps_satisfied(self, sizing_json):
        r = _find_result(sizing_json, "ETH", "risk_constrained", 1.0, eps=0.01)
        assert r["pi_liq"] <= 0.01 + 1e-9

    def test_eth_rc_5pct_eps_satisfied(self, sizing_json):
        r = _find_result(sizing_json, "ETH", "risk_constrained", 1.0, eps=0.05)
        assert r["pi_liq"] <= 0.05 + 1e-9

    def test_btc_rc_5pct_eps_satisfied(self, sizing_json):
        r = _find_result(sizing_json, "BTC", "risk_constrained", 1.0, eps=0.05)
        assert r["pi_liq"] <= 0.05 + 1e-9

    # --- Stress monotonicity: higher stress → more margin ---

    def test_eth_alpha_increases_with_stress_rc_5pct(self, sizing_json):
        r1 = _find_result(sizing_json, "ETH", "risk_constrained", 1.0, eps=0.05)
        r2 = _find_result(sizing_json, "ETH", "risk_constrained", 2.0, eps=0.05)
        assert r2["alpha_star"] >= r1["alpha_star"]

    def test_btc_alpha_increases_with_stress_rc_5pct(self, sizing_json):
        r1 = _find_result(sizing_json, "BTC", "risk_constrained", 1.0, eps=0.05)
        r2 = _find_result(sizing_json, "BTC", "risk_constrained", 2.0, eps=0.05)
        assert r2["alpha_star"] >= r1["alpha_star"]

    # --- Carry sanity: net carry is positive for positive-theta assets ---

    def test_eth_rc_1x_carry_positive(self, sizing_json):
        r = _find_result(sizing_json, "ETH", "risk_constrained", 1.0, eps=0.05)
        assert r["carry_net_hourly"] >= 0.0

    def test_btc_rc_1x_carry_positive(self, sizing_json):
        r = _find_result(sizing_json, "BTC", "risk_constrained", 1.0, eps=0.05)
        assert r["carry_net_hourly"] >= 0.0

    # --- Magnitude bands: alpha* should be well under 1 at base vol ---

    def test_eth_base_vol_alpha_below_50pct(self, sizing_json):
        r = _find_result(sizing_json, "ETH", "risk_constrained", 1.0, eps=0.05)
        assert r["alpha_star"] < 0.5, (
            f"ETH base-vol α*={r['alpha_star']:.3f} unexpectedly high (> 0.5)"
        )

    def test_btc_base_vol_alpha_below_50pct(self, sizing_json):
        r = _find_result(sizing_json, "BTC", "risk_constrained", 1.0, eps=0.05)
        assert r["alpha_star"] < 0.5

    # --- Annualised carry consistency ---

    def test_annual_carry_consistent_with_hourly(self, sizing_json):
        for r in sizing_json["results"]:
            expected = r["carry_net_hourly"] * 8760.0 * 10_000.0
            assert r["carry_annual_bps"] == pytest.approx(expected, rel=1e-6)
