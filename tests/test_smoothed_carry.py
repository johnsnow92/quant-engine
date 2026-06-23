"""Tests for the low-turnover SmoothedFundingCarry signal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.strategies.carry_signal import FundingCarry, SmoothedFundingCarry


def _noisy_funding(n: int = 500, seed: int = 3) -> pd.DataFrame:
    """Persistent positive funding regime with hourly noise that flips the raw sign."""
    rng = np.random.default_rng(seed)
    base = 2e-5  # persistently positive smoothed funding
    noise = rng.normal(0, 3e-5, n)  # noise large enough to flip the instantaneous sign
    return pd.DataFrame({"funding_rate": base + noise})


def test_smoothed_holds_through_noise_and_cuts_turnover():
    market = _noisy_funding()
    naive = FundingCarry(threshold=0.0).generate_carry_positions(market)
    smooth = SmoothedFundingCarry(smooth_window=24, enter_threshold=1e-5).generate_carry_positions(market)

    naive_trades = int(naive.diff().abs().gt(1e-9).sum())
    smooth_trades = int(smooth.diff().abs().gt(1e-9).sum())

    # Smoothing must dramatically reduce churn on a sign-flipping series.
    assert smooth_trades < naive_trades / 5
    # And it should mostly hold the +1 (harvest-positive-funding) leg.
    assert (smooth.iloc[24:] == 1.0).mean() > 0.7


def test_smoothed_flat_when_funding_centered_on_zero():
    rng = np.random.default_rng(1)
    market = pd.DataFrame({"funding_rate": rng.normal(0, 1e-6, 400)})  # tiny, no real regime
    smooth = SmoothedFundingCarry(smooth_window=24, enter_threshold=1e-5).generate_carry_positions(market)
    # Smoothed funding never clears the entry band -> stays flat.
    assert (smooth == 0.0).all()


def test_sign_convention_negative_regime():
    market = pd.DataFrame({"funding_rate": [-3e-5] * 200})
    smooth = SmoothedFundingCarry(smooth_window=24, enter_threshold=1e-5).generate_carry_positions(market)
    assert (smooth.iloc[30:] == -1.0).all()  # persistent negative funding -> short spot/long perp
