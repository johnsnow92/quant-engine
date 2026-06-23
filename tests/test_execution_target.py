"""Tests for the live target-portfolio sizer (execution/target.py)."""
import math

import numpy as np
import pytest

from quant_engine.execution.target import (
    ExecutionConfig,
    compute_sigma_h,
    compute_target,
    quantize_to_contracts,
    BITNOMIAL_CONTRACT_SIZES,
)


def _series(n: int, vol: float, seed: int, start: float = 2000.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0, vol, n)))


def test_compute_sigma_h_matches_numpy():
    prices = [100.0, 110.0, 121.0, 100.0, 105.0]
    expected = float(np.diff(np.log(prices)).std(ddof=1))
    assert compute_sigma_h(prices) == pytest.approx(expected)


def test_compute_sigma_h_needs_two_points():
    with pytest.raises(ValueError):
        compute_sigma_h([100.0])


def test_locked_config_defaults():
    cfg = ExecutionConfig()
    assert cfg.eps == 0.01
    assert cfg.stress_mult == 2.0
    assert cfg.theta_F == 0.01


def test_target_is_delta_neutral():
    prices = _series(800, 0.01, seed=1)
    t = compute_target("ETH", 10_000.0, prices, spot_price=2000.0, perp_price=2000.0)
    # matched quantity → long spot exactly offsets short perp
    assert t.spot_qty == pytest.approx(-t.perp_qty)
    assert t.spot_qty > 0 and t.perp_qty < 0


def test_notional_split_sums_to_equity():
    prices = _series(800, 0.01, seed=2)
    eq = 25_000.0
    t = compute_target("ETH", eq, prices, 2000.0, 2000.0)
    # spot held (1-α)·equity + α·equity posted as margin = equity
    assert t.spot_notional + t.perp_margin == pytest.approx(eq)
    assert t.spot_notional == pytest.approx((1 - t.alpha) * eq)


def test_alpha_in_unit_interval():
    prices = _series(800, 0.015, seed=3)
    t = compute_target("BTC", 5_000.0, prices, 60000.0, 60000.0)
    assert 0.0 <= t.alpha < 1.0


def test_higher_vol_gives_higher_alpha():
    lo = compute_target("ETH", 10_000.0, _series(800, 0.004, seed=4), 2000.0, 2000.0)
    hi = compute_target("ETH", 10_000.0, _series(800, 0.020, seed=4), 2000.0, 2000.0)
    assert hi.alpha > lo.alpha          # more vol → more margin to hold the barrier


def test_barrier_above_entry():
    prices = _series(800, 0.012, seed=5)
    t = compute_target("ETH", 10_000.0, prices, 2000.0, 2000.0)
    assert t.r_liq > 1.0
    assert t.barrier_move_pct > 0.0


def test_pi_liq_within_eps():
    prices = _series(800, 0.012, seed=6)
    cfg = ExecutionConfig(eps=0.01)
    t = compute_target("ETH", 10_000.0, prices, 2000.0, 2000.0, cfg)
    assert t.pi_liq <= cfg.eps + 1e-9   # α sized so liquidation prob ≤ ε


def test_perp_notional_tracks_spot_when_prices_equal():
    prices = _series(800, 0.01, seed=7)
    t = compute_target("ETH", 10_000.0, prices, 2000.0, 2000.0)
    assert t.perp_notional == pytest.approx(t.spot_notional)


def test_perp_notional_uses_perp_price_on_basis():
    prices = _series(800, 0.01, seed=8)
    t = compute_target("ETH", 10_000.0, prices, spot_price=2000.0, perp_price=2010.0)
    # quantity matched to spot; perp notional marked at the (higher) perp price
    assert t.perp_notional == pytest.approx(t.spot_qty * 2010.0)
    assert t.perp_notional > t.spot_notional


def test_rejects_nonpositive_equity():
    prices = _series(800, 0.01, seed=9)
    with pytest.raises(ValueError):
        compute_target("ETH", 0.0, prices, 2000.0, 2000.0)


def test_rejects_nonpositive_price():
    prices = _series(800, 0.01, seed=10)
    with pytest.raises(ValueError):
        compute_target("ETH", 10_000.0, prices, 0.0, 2000.0)


# --- contract quantization (Bitnomial US perps trade in whole contracts) -------

def test_quantize_snaps_perp_to_whole_contracts():
    t = compute_target("ETH", 10_000.0, _series(800, 0.01, seed=11), 2000.0, 2000.0)
    q = quantize_to_contracts(t, 0.5)               # ETH = 0.5 per contract
    n = abs(q.perp_qty) / 0.5
    assert n == pytest.approx(round(n))             # exact whole number of contracts
    assert abs(q.perp_qty) % 0.5 == pytest.approx(0.0, abs=1e-9)


def test_quantize_keeps_delta_neutral():
    t = compute_target("BTC", 50_000.0, _series(800, 0.012, seed=12), 60000.0, 60000.0)
    q = quantize_to_contracts(t, 0.01)
    assert q.spot_qty == pytest.approx(-q.perp_qty)  # matched → delta-neutral after snapping


def test_quantize_residual_small_at_reasonable_equity():
    t = compute_target("ETH", 25_000.0, _series(800, 0.01, seed=13), 2000.0, 2000.0)
    q = quantize_to_contracts(t, 0.5)
    # snapped spot notional within one contract's notional of the ideal
    assert abs(q.spot_notional - t.spot_notional) <= 0.5 * t.spot_price + 1e-6


def test_quantize_rejects_bad_contract_size():
    t = compute_target("ETH", 10_000.0, _series(800, 0.01, seed=14), 2000.0, 2000.0)
    with pytest.raises(ValueError):
        quantize_to_contracts(t, 0.0)


def test_contract_sizes_present():
    assert BITNOMIAL_CONTRACT_SIZES["BTC"] == 0.01
    assert BITNOMIAL_CONTRACT_SIZES["ETH"] == 0.5
    assert BITNOMIAL_CONTRACT_SIZES["SOL"] == 5.0
