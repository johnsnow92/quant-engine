"""Moving-average crossover — a trivial trend strategy to prove the pipeline.

Long when the fast MA is above the slow MA, short otherwise. Flat during the
warmup window before the slow MA has enough observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MACrossover:
    fast: int = 12
    slow: int = 48
    name: str = "ma_crossover"

    def generate_positions(self, market: pd.DataFrame) -> pd.Series:
        if self.fast >= self.slow:
            raise ValueError("fast window must be smaller than slow window")
        close = market["close"]
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()
        position = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=market.index)
        position[slow_ma.isna()] = 0.0
        return position.rename("position")
