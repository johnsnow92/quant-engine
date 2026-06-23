"""Funding carry signal — picks the carry direction from the funding rate.

The returned ``carry_sign`` is consumed by ``backtest.carry.run_carry_backtest``,
which owns the two-leg PnL:

    s = +1  -> short perp + long spot   (harvest positive funding)
    s = -1  -> long perp + short spot   (harvest negative funding)
    s =  0  -> flat
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FundingCarry:
    threshold: float = 0.0
    name: str = "funding_carry"

    def generate_carry_positions(self, market: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in market.columns:
            raise ValueError("market frame missing 'funding_rate' column")
        funding = market["funding_rate"].fillna(0.0)
        sign = pd.Series(0.0, index=market.index)
        sign[funding > self.threshold] = 1.0
        sign[funding < -self.threshold] = -1.0
        return sign.rename("carry_sign")


@dataclass
class SmoothedFundingCarry:
    """Low-turnover carry: smooth funding, then a hysteresis state machine.

    Naive sign-following flips on every funding wiggle and bleeds the (small)
    funding edge to fees. This harvester smooths funding over ``smooth_window``
    bars and only opens when the smoothed rate clears ``enter_threshold``,
    holding the leg until it decays back inside ``exit_threshold`` (toward 0).
    The dead band between ``exit`` and ``enter`` is what suppresses churn, so the
    book trades a handful of times per persistent funding regime instead of
    hundreds of times per month.

    Sign convention matches ``run_carry_backtest``: +1 short perp + long spot
    (harvest positive funding), -1 long perp + short spot.
    """

    smooth_window: int = 24       # bars (hours) to average funding over
    enter_threshold: float = 1e-5  # smoothed funding/hr magnitude to open
    exit_threshold: float = 0.0    # smoothed funding/hr to close back toward flat
    name: str = "smoothed_funding_carry"

    def generate_carry_positions(self, market: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in market.columns:
            raise ValueError("market frame missing 'funding_rate' column")
        sm = market["funding_rate"].fillna(0.0).rolling(self.smooth_window).mean().to_numpy()
        pos = [0.0] * len(sm)
        state = 0.0
        for i, s in enumerate(sm):
            if s != s:  # NaN during the warmup window
                pos[i] = state
                continue
            if state == 0.0:
                if s >= self.enter_threshold:
                    state = 1.0
                elif s <= -self.enter_threshold:
                    state = -1.0
            elif state == 1.0:
                if s <= -self.enter_threshold:
                    state = -1.0
                elif s <= self.exit_threshold:
                    state = 0.0
            elif state == -1.0:
                if s >= self.enter_threshold:
                    state = 1.0
                elif s >= -self.exit_threshold:
                    state = 0.0
            pos[i] = state
        return pd.Series(pos, index=market.index, name="carry_sign")
