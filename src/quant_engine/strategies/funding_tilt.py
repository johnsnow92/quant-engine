"""Funding-rate carry tilt — a derivatives-native trivial strategy.

Perpetual longs pay funding when the rate is positive, so this strategy goes
short to collect funding when funding is positive and long when it is negative.
The actual funding PnL is modeled by the backtest engine via the per-bar
funding leg; this module only decides direction.

Known simplification: this is a directional tilt on the perp, not a
market-neutral perp/spot carry. A true carry trade hedges the price leg with
spot. Treat the price exposure here as the next thing to neutralize.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FundingTilt:
    threshold: float = 0.0
    name: str = "funding_tilt"

    def generate_positions(self, market: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in market.columns:
            raise ValueError("market frame missing 'funding_rate' column")
        funding = market["funding_rate"].fillna(0.0)
        position = pd.Series(0.0, index=market.index)
        position[funding > self.threshold] = -1.0
        position[funding < -self.threshold] = 1.0
        return position.rename("position")
