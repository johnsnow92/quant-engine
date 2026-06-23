"""Tests for the carry strategy backtest (analysis/backtest.py).

Three layers:
  1. Unit tests — merge_price_funding alignment.
  2. Simulation tests — known synthetic inputs produce known outputs.
  3. Regression locks — load data/backtest_results.json and assert key bands.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_engine.analysis.backtest import (
    BacktestConfig,
    merge_price_funding,
    run_backtest,
    results_to_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_JSON = REPO_ROOT / "data" / "backtest_results.json"

# Short warm-up for fast synthetic tests (7d × 24h = 168h)
FAST_CONFIG = BacktestConfig(
    vol_lookback_days=7,
    carry_lookback_days=7,
    rebal_hours=24,
    eps=0.05,
    stress_mult=1.0,
    lgd=1.0,
    theta_F=0.01,
    h_hours=24.0,
    fee_bps_per_side=0.0,   # zero-cost: these tests isolate strategy mechanics
    funding_capture=1.0,    # cost behavior is covered by TestCostModel
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dfs(
    n_hours: int = 400,
    price_start: float = 2000.0,
    price_growth_per_hour: float = 0.0,
    funding_rate: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = np.arange(n_hours, dtype=float) * 3_600_000
    prices = price_start * (1.0 + price_growth_per_hour) ** np.arange(n_hours)
    price_df = pd.DataFrame({"ts": ts, "close": prices})
    funding_df = pd.DataFrame({"ts": ts, "funding_rate": np.full(n_hours, funding_rate)})
    return price_df, funding_df


# ---------------------------------------------------------------------------
# 1. Unit tests — merge_price_funding
# ---------------------------------------------------------------------------

class TestMergePriceFunding:
    def test_aligns_on_hour_bucket(self):
        ts_base = np.arange(5) * 3_600_000
        price_df = pd.DataFrame({"ts": ts_base + 60_000, "close": np.ones(5) * 100})
        funding_df = pd.DataFrame({"ts": ts_base, "funding_rate": np.full(5, 1e-4)})
        merged = merge_price_funding(price_df, funding_df)
        assert len(merged) == 5

    def test_inner_join_drops_unmatched(self):
        ts_p = np.array([0, 3_600_000, 7_200_000], dtype=float)
        ts_f = np.array([3_600_000, 7_200_000, 10_800_000], dtype=float)
        price_df = pd.DataFrame({"ts": ts_p, "close": [1.0, 2.0, 3.0]})
        funding_df = pd.DataFrame({"ts": ts_f, "funding_rate": [0.01, 0.02, 0.03]})
        merged = merge_price_funding(price_df, funding_df)
        assert len(merged) == 2

    def test_sorted_by_hour_ms(self):
        n = 10
        ts = np.arange(n) * 3_600_000
        price_df = pd.DataFrame({"ts": ts[::-1], "close": np.ones(n)})
        funding_df = pd.DataFrame({"ts": ts, "funding_rate": np.zeros(n)})
        merged = merge_price_funding(price_df, funding_df)
        assert list(merged["hour_ms"]) == sorted(merged["hour_ms"].tolist())

    def test_deduplicates_price(self):
        ts = np.array([0.0, 0.0, 3_600_000.0])
        price_df = pd.DataFrame({"ts": ts, "close": [1.0, 2.0, 3.0]})
        funding_df = pd.DataFrame({"ts": [0.0, 3_600_000.0], "funding_rate": [1e-4, 2e-4]})
        merged = merge_price_funding(price_df, funding_df)
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# 2. Simulation tests
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_insufficient_data_raises(self):
        price_df, funding_df = _make_dfs(n_hours=50)
        with pytest.raises(ValueError, match="Insufficient data"):
            run_backtest(price_df, funding_df, "X", FAST_CONFIG)

    def test_flat_price_positive_funding_earns_carry(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert result.total_return > 0
        assert result.liq_count == 0

    def test_zero_funding_flat_equity(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=0.0)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert abs(result.total_return) < 1e-9
        assert result.gross_carry == pytest.approx(0.0, abs=1e-12)

    def test_warmup_equity_stays_at_one(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        warm_up = FAST_CONFIG.vol_lookback_hours
        assert np.all(result.equity_series[:warm_up] == pytest.approx(1.0, abs=1e-12))

    def test_no_alpha_during_warmup(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        warm_up = FAST_CONFIG.vol_lookback_hours
        assert np.all(result.alpha_series[:warm_up] == 0.0)

    def test_equity_nonnegative(self):
        price_df, funding_df = _make_dfs(n_hours=500, funding_rate=5e-5)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert np.all(result.equity_series >= 0)

    def _spike_dfs(self, n: int = 400, spike_mult: float = 5.0):
        """Price path with ~1%/hr baseline vol (so α sizes above α_crit) plus a
        large upward spike that pushes r_spot well past r_liq."""
        warm_up = FAST_CONFIG.vol_lookback_hours
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0.0, 0.01, n)
        prices = 2000.0 * np.exp(np.cumsum(log_rets))
        spike_idx = warm_up + 50
        prices[spike_idx] = prices[spike_idx - 1] * spike_mult  # far above r_liq
        ts = np.arange(n, dtype=float) * 3_600_000
        price_df = pd.DataFrame({"ts": ts, "close": prices})
        funding_df = pd.DataFrame({"ts": ts, "funding_rate": np.full(n, 1e-4)})
        return price_df, funding_df, spike_idx

    def test_price_spike_causes_liquidation(self):
        price_df, funding_df, _ = self._spike_dfs()
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert result.liq_count >= 1
        assert result.liq_losses > 0

    def test_liquidation_reduces_equity(self):
        price_df, funding_df, spike_idx = self._spike_dfs()
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        # equity right after the spike must be below the pre-spike level
        assert result.equity_series[spike_idx + 1] < result.equity_series[spike_idx - 1]

    def test_tighter_eps_lower_carry(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        cfg_loose = BacktestConfig(**{**FAST_CONFIG.__dict__, "eps": 0.10})
        cfg_tight = BacktestConfig(**{**FAST_CONFIG.__dict__, "eps": 0.01})
        r_loose = run_backtest(price_df, funding_df, "X", cfg_loose)
        r_tight = run_backtest(price_df, funding_df, "X", cfg_tight)
        assert r_tight.gross_carry <= r_loose.gross_carry + 1e-9

    def test_net_pnl_equals_total_return(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert result.net_pnl == pytest.approx(result.total_return, rel=1e-6)

    def test_alpha_in_unit_interval_post_warmup(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        warm_up = FAST_CONFIG.vol_lookback_hours
        active = result.alpha_series[warm_up:]
        assert np.all(active >= 0.0)
        assert np.all(active <= 1.0)

    def test_sharpe_positive_for_strong_carry(self):
        price_df, funding_df = _make_dfs(n_hours=500, funding_rate=2e-4)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert result.sharpe > 0

    def test_result_dimensions_consistent(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=5e-5)
        result = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        assert result.n_hours == 400
        assert len(result.equity_series) == 400
        assert len(result.alpha_series) == 400
        assert len(result.timestamps_ms) == 400
        assert result.n_active_hours == 400 - FAST_CONFIG.vol_lookback_hours

    def test_summary_contains_asset_name(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=5e-5)
        result = run_backtest(price_df, funding_df, "MYASSET", FAST_CONFIG)
        assert "MYASSET" in result.summary()

    def test_results_to_table_has_header(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=5e-5)
        r = run_backtest(price_df, funding_df, "X", FAST_CONFIG)
        table = results_to_table([r])
        assert "AnnRet" in table and "Sharpe" in table


# ---------------------------------------------------------------------------
# 2b. Cost model — fees + funding-capture haircut
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> BacktestConfig:
    return BacktestConfig(**{**FAST_CONFIG.__dict__, **overrides})


class TestCostModel:
    def test_zero_cost_has_no_fees_or_drag(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r = run_backtest(price_df, funding_df, "X", FAST_CONFIG)  # zero-cost
        assert r.fees == pytest.approx(0.0, abs=1e-12)
        assert r.funding_drag == pytest.approx(0.0, abs=1e-12)

    def test_fees_charged_when_fee_bps_positive(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=4.5))
        assert r.fees > 0

    def test_fees_reduce_total_return(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r_free = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=0.0))
        r_fee = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=4.5))
        assert r_fee.total_return < r_free.total_return

    def test_higher_fees_lower_return(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r_lo = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=2.0))
        r_hi = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=10.0))
        assert r_hi.fees > r_lo.fees
        assert r_hi.total_return < r_lo.total_return

    def test_funding_haircut_creates_drag(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r = run_backtest(price_df, funding_df, "X", _cfg(funding_capture=0.90))
        assert r.funding_drag > 0

    def test_funding_drag_proportional_to_haircut(self):
        # drag ≈ (1−capture) × gross_carry
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r = run_backtest(price_df, funding_df, "X", _cfg(funding_capture=0.90))
        assert r.funding_drag == pytest.approx(0.10 * r.gross_carry, rel=1e-6)

    def test_zero_funding_with_fees_is_negative(self):
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=0.0)
        r = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=4.5))
        assert r.fees > 0
        assert r.total_return < 0

    def test_decomposition_consistency(self):
        # net_pnl ≈ gross_carry − funding_drag − fees − liq_losses (compounding residual small)
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r = run_backtest(price_df, funding_df, "X", _cfg(fee_bps_per_side=4.5, funding_capture=0.90))
        recomposed = r.gross_carry - r.funding_drag - r.fees - r.liq_losses
        assert r.net_pnl == pytest.approx(recomposed, abs=1e-3)

    def test_haircut_lowers_net_return(self):
        # A bigger funding-capture haircut reduces the net return (more drag).
        price_df, funding_df = _make_dfs(n_hours=400, funding_rate=1e-4)
        r_full = run_backtest(price_df, funding_df, "X", _cfg(funding_capture=1.0))
        r_hair = run_backtest(price_df, funding_df, "X", _cfg(funding_capture=0.80))
        assert r_hair.funding_drag > r_full.funding_drag
        assert r_hair.total_return < r_full.total_return


# ---------------------------------------------------------------------------
# 3. Regression locks — live backtest results (data/backtest_results.json)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bt_json() -> dict:
    if not BACKTEST_JSON.exists():
        pytest.skip("backtest_results.json not found — run run_backtest.py first")
    with open(BACKTEST_JSON) as f:
        return json.load(f)


def _find_bt(data: dict, asset: str, eps: float, stress_mult: float) -> dict:
    for r in data["results"]:
        if (r["asset"] == asset
                and abs(r["eps"] - eps) < 1e-9
                and abs(r["stress_mult"] - stress_mult) < 1e-9):
            return r
    pytest.skip(f"No result for {asset}/ε={eps}/stress={stress_mult}")


class TestRegressionLocks:
    """Assert live backtest results stay within expected bands.

    Canonical run: Coinbase spot prices over the full ~400-day window (a complete
    funding cycle) + Hyperliquid funding, with perp-margin re-centering and the
    cost model on. Over this fuller window — which contains far larger price
    moves than the recent calm ~178-day window — the safe frontier is TIGHTER:
    2× stress is NOT universally zero-liq. The robust universal config that stays
    zero-liquidation for BOTH assets is ε=1% + 2× vol stress, earning ~+6.2-6.5%
    net of costs (the rich 2025 funding regime). Looser ε or lower stress get
    liquidated by the big moves and turn negative.
    """

    def test_results_nonempty(self, bt_json):
        assert len(bt_json["results"]) > 0

    def test_price_source_is_coinbase(self, bt_json):
        # Canonical run uses Coinbase deep history for the full funding cycle.
        assert bt_json.get("price_source") == "coinbase"

    def test_both_assets_present(self, bt_json):
        found = {r["asset"] for r in bt_json["results"]}
        assert {"ETH", "BTC"}.issubset(found)

    def test_gross_carry_positive_everywhere(self, bt_json):
        # Funding has been net-positive — gross carry is positive in every config.
        for r in bt_json["results"]:
            assert r["gross_carry"] > 0, f"{r['asset']} ε={r['eps']} {r['stress_mult']}×"

    def test_eps1_2x_zero_liquidations_both_assets(self, bt_json):
        # The robust winner over the full regime: tightest ε + 2× stress keeps the
        # barrier above even the big moves → zero liquidations for both assets.
        for asset in ("ETH", "BTC"):
            r = _find_bt(bt_json, asset, 0.01, 2.0)
            assert r["liq_count"] == 0, (
                f"{asset} ε=1% 2× had {r['liq_count']} liquidations"
            )
            assert r["liq_losses"] == pytest.approx(0.0, abs=1e-12)

    def test_eps1_2x_positive_net_both_assets(self, bt_json):
        # Net of fees + funding-spread drag, the zero-liq ε=1%/2× config is
        # clearly positive over the full funding cycle (~+6%/yr).
        for asset in ("ETH", "BTC"):
            r = _find_bt(bt_json, asset, 0.01, 2.0)
            assert r["net_pnl"] > 0, f"{asset} ε=1% 2× net={r['net_pnl']}"
            assert r["ann_return"] > 0

    def test_net_below_gross_in_safe_config(self, bt_json):
        # In the zero-liq ε=1%/2× config, net < gross — the gap is funding drag + fees.
        for asset in ("ETH", "BTC"):
            r = _find_bt(bt_json, asset, 0.01, 2.0)
            assert r["net_pnl"] < r["gross_carry"]
            assert r["fees"] > 0 and r["funding_drag"] > 0

    def test_higher_stress_monotone_fewer_liquidations(self, bt_json):
        # More vol stress → more margin → higher barrier → never MORE liquidations.
        for asset in ("ETH", "BTC"):
            for eps in (0.01, 0.05, 0.10):
                liq_1x = _find_bt(bt_json, asset, eps, 1.0)["liq_count"]
                liq_15 = _find_bt(bt_json, asset, eps, 1.5)["liq_count"]
                liq_2x = _find_bt(bt_json, asset, eps, 2.0)["liq_count"]
                assert liq_1x >= liq_15 >= liq_2x, (
                    f"{asset} ε={eps:.0%} non-monotone liq: {liq_1x},{liq_15},{liq_2x}"
                )

    def test_1x_stress_unsafe_at_loose_eps(self, bt_json):
        # The danger zone: at 1× stress + loose ε the barrier sits inside the move
        # distribution and the position is liquidated many times over the cycle.
        for asset in ("ETH", "BTC"):
            r = _find_bt(bt_json, asset, 0.10, 1.0)
            assert r["liq_count"] >= 10, f"{asset} ε=10% 1× liq={r['liq_count']}"
            assert r["ann_return"] < 0

    def test_tighter_eps_lower_carry_eth_safe_configs(self, bt_json):
        # ETH ε=1% and ε=5% are both zero-liq at 2× stress, so comparing their
        # gross carry is clean: tighter ε → bigger α → less spot → lower gross.
        r_tight = _find_bt(bt_json, "ETH", 0.01, 2.0)
        r_loose = _find_bt(bt_json, "ETH", 0.05, 2.0)
        assert r_tight["gross_carry"] <= r_loose["gross_carry"] + 1e-6

    def test_equity_daily_starts_at_one(self, bt_json):
        for r in bt_json["results"]:
            assert r["equity_daily"][0] == pytest.approx(1.0, abs=1e-9)

    def test_equity_daily_nonnegative(self, bt_json):
        for r in bt_json["results"]:
            assert all(e >= 0 for e in r["equity_daily"])

    def test_alpha_daily_in_unit_interval(self, bt_json):
        for r in bt_json["results"]:
            assert all(0.0 <= a <= 1.0 for a in r["alpha_daily"])

    def test_active_hours_full_cycle(self, bt_json):
        # Coinbase serves the full ~400d hourly history; after 30d warm-up the
        # active window is ~369 days ≈ 8,800h. Allow a band for candle gaps.
        for r in bt_json["results"]:
            assert 8000 <= r["n_active_hours"] <= 9600, (
                f"{r['asset']} n_active_hours={r['n_active_hours']}"
            )

    def test_costs_present_and_nonnegative(self, bt_json):
        # The live run uses the default cost model — fees and funding drag are
        # serialized and never negative.
        for r in bt_json["results"]:
            assert r["fees"] >= 0
            assert r["funding_drag"] >= 0
