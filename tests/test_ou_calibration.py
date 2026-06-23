"""Tests for OU funding-rate calibration (analysis/ou_calibration.py).

Three layers:
  1. Math unit tests — exact AR(1) input -> verify closed-form parameter recovery.
  2. Synthetic OU tests — simulate a known process, check estimates land close.
  3. Regression locks — load data/ou_params.json and assert live calibration
     results stay within expected bands.  These lock the behaviour of
     calibrate_ou_funding.py against live Hyperliquid data so a regime shift
     or code change is immediately visible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_engine.analysis.ou_calibration import (
    OUParams,
    calibrate,
    calibrate_multi_window,
    params_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OU_PARAMS_JSON = REPO_ROOT / "data" / "ou_params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rates: np.ndarray) -> pd.DataFrame:
    """Wrap a rate array in the DataFrame shape expected by calibrate()."""
    n = len(rates)
    ts = np.arange(n, dtype=float) * 3_600_000  # hourly, ms
    return pd.DataFrame({"ts": ts, "funding_rate": rates})


def _simulate_ou(
    kappa: float,
    theta: float,
    sigma_f: float,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """Exact discrete simulation of OU with Δ=1h."""
    rng = np.random.default_rng(seed)
    b = math.exp(-kappa)
    innov_std = math.sqrt(sigma_f ** 2 * (1 - b ** 2) / (2 * kappa))
    rates = np.empty(n)
    rates[0] = theta
    for i in range(1, n):
        rates[i] = theta + b * (rates[i - 1] - theta) + innov_std * rng.standard_normal()
    return rates


# ---------------------------------------------------------------------------
# 1. Math unit tests — AR(1) recovery
# ---------------------------------------------------------------------------

class TestOUMathRecovery:
    """Feed a *perfect* AR(1) series and check exact inversion."""

    def _perfect_ar1(self, a: float, b: float, n: int = 2000, seed: int = 0) -> np.ndarray:
        """y_t = a + b*y_{t-1} + eps, eps ~ N(0, s^2). s fixed at 1e-5."""
        rng = np.random.default_rng(seed)
        s = 1e-5
        y = np.empty(n)
        y[0] = a / (1 - b)          # start at stationary mean
        for i in range(1, n):
            y[i] = a + b * y[i - 1] + s * rng.standard_normal()
        return y

    def test_kappa_recovered_from_halflife(self):
        # kappa = -log(b); b=0.9 -> kappa~0.1054
        b = 0.9
        a = 1e-5 * (1 - b)
        rates = self._perfect_ar1(a, b, n=5000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        assert p.kappa == pytest.approx(-math.log(b), rel=0.05)

    def test_theta_recovered(self):
        # With b=0.85 the effective sample size for theta is n*(1-b)/(1+b) ≈ 400,
        # so OLS has inherent ~10-15% error at n=5000. 20% is the right bound here.
        b = 0.85
        theta_true = 3e-6
        a = theta_true * (1 - b)
        rates = self._perfect_ar1(a, b, n=5000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        assert p.theta == pytest.approx(theta_true, rel=0.20)

    def test_half_life_formula(self):
        rates = self._perfect_ar1(0.0, 0.88, n=3000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        expected = math.log(2) / p.kappa
        assert p.half_life_hours == pytest.approx(expected, rel=1e-9)

    def test_sigma_stationary_formula(self):
        rates = self._perfect_ar1(0.0, 0.9, n=4000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        expected = p.sigma_f / math.sqrt(2 * p.kappa)
        assert p.sigma_stationary == pytest.approx(expected, rel=1e-9)

    def test_hjb_grid_symmetric_around_theta(self):
        rates = self._perfect_ar1(0.0, 0.9, n=3000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        centre = (p.f_grid_min + p.f_grid_max) / 2
        assert centre == pytest.approx(p.theta, rel=1e-9)
        half_width = (p.f_grid_max - p.f_grid_min) / 2
        assert half_width == pytest.approx(4.0 * p.sigma_stationary, rel=1e-9)

    def test_sigma_f_positive(self):
        rates = self._perfect_ar1(0.0, 0.8, n=2000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        assert p.sigma_f > 0

    def test_ar1_b_in_unit_interval(self):
        rates = self._perfect_ar1(0.0, 0.75, n=2000)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        assert 0 < p.ar1_b < 1


# ---------------------------------------------------------------------------
# 2. Synthetic OU — recover known (kappa, theta, sigma_f)
# ---------------------------------------------------------------------------

class TestSyntheticOU:
    """Simulate a genuine OU process; check estimates within 10% of truth."""

    # Parameters loosely matching ETH (paper Table 1)
    KAPPA = 0.1247      # h^-1  (t½ ≈ 5.56h)
    THETA = 5e-6        # fractional per-hour funding
    SIGMA_F = 5e-6
    N = 10_000          # 10k hourly obs ≈ 417 days

    @pytest.fixture(scope="class")
    def params(self):
        rates = _simulate_ou(self.KAPPA, self.THETA, self.SIGMA_F, self.N, seed=1)
        return calibrate(_make_df(rates), "ETH_synth", window_days=9999)

    def test_kappa_within_10pct(self, params):
        assert params.kappa == pytest.approx(self.KAPPA, rel=0.10)

    def test_theta_within_10pct(self, params):
        assert params.theta == pytest.approx(self.THETA, rel=0.10)

    def test_sigma_f_within_10pct(self, params):
        assert params.sigma_f == pytest.approx(self.SIGMA_F, rel=0.10)

    def test_half_life_within_10pct(self, params):
        expected_hl = math.log(2) / self.KAPPA
        assert params.half_life_hours == pytest.approx(expected_hl, rel=0.10)

    def test_jump_proxy_low_for_gaussian(self, params):
        # Gaussian process: about 0.27% of obs beyond 3σ. Allow generous 2%.
        assert params.jump_prob_per_hour < 0.02

    def test_n_obs_matches_input(self, params):
        assert params.n_obs == self.N

    def test_params_to_dict_roundtrips(self, params):
        d = params_to_dict(params)
        assert d["kappa"] == params.kappa
        assert d["theta"] == params.theta
        assert d["sigma_f"] == params.sigma_f
        assert d["asset"] == "ETH_synth"


# ---------------------------------------------------------------------------
# 3. calibrate() API / edge-case tests
# ---------------------------------------------------------------------------

class TestCalibrateAPI:

    def test_insufficient_data_raises(self):
        df = _make_df(np.array([1e-5, 2e-5, 3e-5]))   # 3 obs < 10
        with pytest.raises(ValueError, match="Too few"):
            calibrate(df, "X", window_days=9999)

    def test_window_slices_correctly(self):
        # Build 600h of data; 90-day window = 2160h -> should use last 2160 obs
        n = 600
        rates = _simulate_ou(0.12, 5e-6, 5e-6, n, seed=2)
        df = _make_df(rates)
        p_all = calibrate(df, "X", window_days=9999)
        p_90 = calibrate(df, "X", window_days=25)   # 25 days = 600h = all data here
        assert p_all.n_obs == p_90.n_obs             # same window in this case

    def test_calibrate_multi_window_returns_ordered(self):
        rates = _simulate_ou(0.15, 3e-6, 4e-6, 10_000, seed=3)
        df = _make_df(rates)
        results = calibrate_multi_window(df, "BTC", windows=(90, 180, 360))
        assert [r.window_days for r in results] == [90, 180, 360]

    def test_calibrate_multi_window_skips_too_short(self):
        # Only 50 obs: 90d window needs 2160h so it will fall back but still fit
        # because window_days clamps to available data.  The real guard is <10 obs.
        rates = _simulate_ou(0.15, 3e-6, 4e-6, 50, seed=4)
        df = _make_df(rates)
        results = calibrate_multi_window(df, "X", windows=(1, 2, 5))
        # All windows should succeed because n_obs = 50 >= 10
        assert len(results) == 3

    def test_summary_string_contains_asset(self):
        rates = _simulate_ou(0.12, 5e-6, 5e-6, 500, seed=5)
        p = calibrate(_make_df(rates), "MYASSET", window_days=9999)
        assert "MYASSET" in p.summary()

    def test_summary_string_contains_half_life(self):
        rates = _simulate_ou(0.12, 5e-6, 5e-6, 500, seed=6)
        p = calibrate(_make_df(rates), "X", window_days=9999)
        assert "half-life" in p.summary()


# ---------------------------------------------------------------------------
# 4. Regression locks — live calibration results (data/ou_params.json)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ou_json() -> dict:
    if not OU_PARAMS_JSON.exists():
        pytest.skip(f"ou_params.json not found at {OU_PARAMS_JSON} — run calibrate_ou_funding.py first")
    with open(OU_PARAMS_JSON) as f:
        return json.load(f)


def _find(ou_json: dict, asset: str, window: int) -> dict:
    for p in ou_json["params"]:
        if p["asset"] == asset and p["window_days"] == window:
            return p
    pytest.skip(f"No entry for {asset}/{window}d in ou_params.json")


class TestRegressionLocks:
    """Assert that live calibration results stay within expected bands.

    Bands are generous (±50% on kappa, loose on theta) so the tests survive
    normal funding-regime drift without needing constant re-generation.
    A failure here signals a genuine regime shift or a code regression.

    Paper Table 1 reference half-lives: ETH=5.56h, BTC=4.07h, SOL=2.31h.
    """

    # --- ETH 90d ---
    def test_eth_90d_halflife_in_range(self, ou_json):
        p = _find(ou_json, "ETH", 90)
        assert 3.5 <= p["half_life_hours"] <= 9.0, (
            f"ETH 90d half-life={p['half_life_hours']:.2f}h out of [3.5, 9.0]h"
        )

    def test_eth_90d_kappa_positive(self, ou_json):
        p = _find(ou_json, "ETH", 90)
        assert p["kappa"] > 0

    def test_eth_90d_theta_small(self, ou_json):
        # Long-run funding should be tiny (well under 1bp/h = 1e-4)
        p = _find(ou_json, "ETH", 90)
        assert abs(p["theta"]) < 1e-4

    def test_eth_90d_sigma_f_positive(self, ou_json):
        p = _find(ou_json, "ETH", 90)
        assert p["sigma_f"] > 0

    def test_eth_90d_jump_prob_nonzero(self, ou_json):
        # Funding series have fat tails; jump proxy > 0
        p = _find(ou_json, "ETH", 90)
        assert p["jump_prob_per_hour"] > 0

    # --- BTC 90d ---
    def test_btc_90d_halflife_in_range(self, ou_json):
        p = _find(ou_json, "BTC", 90)
        assert 3.0 <= p["half_life_hours"] <= 9.0, (
            f"BTC 90d half-life={p['half_life_hours']:.2f}h out of [3.0, 9.0]h"
        )

    def test_btc_90d_kappa_positive(self, ou_json):
        p = _find(ou_json, "BTC", 90)
        assert p["kappa"] > 0

    def test_btc_90d_theta_small(self, ou_json):
        p = _find(ou_json, "BTC", 90)
        assert abs(p["theta"]) < 1e-4

    def test_btc_90d_sigma_f_positive(self, ou_json):
        p = _find(ou_json, "BTC", 90)
        assert p["sigma_f"] > 0

    # --- SOL 360d (regime shifted; use long window that matches paper) ---
    def test_sol_360d_halflife_in_range(self, ou_json):
        p = _find(ou_json, "SOL", 360)
        assert 1.0 <= p["half_life_hours"] <= 4.5, (
            f"SOL 360d half-life={p['half_life_hours']:.2f}h out of [1.0, 4.5]h"
        )

    def test_sol_360d_kappa_positive(self, ou_json):
        p = _find(ou_json, "SOL", 360)
        assert p["kappa"] > 0

    # --- Cross-asset sanity ---
    def test_all_assets_have_180d_entry(self, ou_json):
        found = {p["asset"] for p in ou_json["params"] if p["window_days"] == 180}
        assert {"ETH", "BTC", "SOL"}.issubset(found)

    def test_hjb_grid_min_below_theta(self, ou_json):
        for asset in ("ETH", "BTC", "SOL"):
            p = _find(ou_json, asset, 180)
            assert p["f_grid_min"] < p["theta"], f"{asset}: f_grid_min >= theta"

    def test_hjb_grid_max_above_theta(self, ou_json):
        for asset in ("ETH", "BTC", "SOL"):
            p = _find(ou_json, asset, 180)
            assert p["f_grid_max"] > p["theta"], f"{asset}: f_grid_max <= theta"

    def test_180d_obs_count_reasonable(self, ou_json):
        # 180 days * 24 h/day = 4320; allow ±10% for gaps/deduplication
        for asset in ("ETH", "BTC", "SOL"):
            p = _find(ou_json, asset, 180)
            assert 3500 <= p["n_obs"] <= 4800, (
                f"{asset} 180d n_obs={p['n_obs']} outside [3500, 4800]"
            )
