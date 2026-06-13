"""Strategy interface.

A strategy maps a market frame to a target position series in [-1, 1], one value
per bar, indexed like the input. Position sizing, funding accrual, fees, and
look-ahead protection are handled by the backtest engine, not the strategy.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def generate_positions(self, market: pd.DataFrame) -> pd.Series:
        """Return target position per bar in [-1, 1], indexed like ``market``."""
        ...
