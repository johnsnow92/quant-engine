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
