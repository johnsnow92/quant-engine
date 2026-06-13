import numpy as np
import pandas as pd

from quant_engine.strategies.funding_tilt import FundingTilt
from quant_engine.strategies.ma_crossover import MACrossover


def test_ma_crossover_positions_bounded_and_warmup_flat():
    market = pd.DataFrame({"close": np.linspace(100, 200, 100)})
    pos = MACrossover(fast=5, slow=20).generate_positions(market)
    assert set(pos.unique()).issubset({-1.0, 0.0, 1.0})
    assert (pos.iloc[:19] == 0.0).all()  # flat before slow window has 20 obs
    assert len(pos) == len(market)


def test_ma_crossover_long_in_uptrend():
    market = pd.DataFrame({"close": np.linspace(100, 200, 100)})
    pos = MACrossover(fast=5, slow=20).generate_positions(market)
    assert pos.iloc[-1] == 1.0


def test_funding_tilt_direction():
    market = pd.DataFrame(
        {"close": [100, 100, 100], "funding_rate": [0.001, -0.001, 0.0]}
    )
    pos = FundingTilt().generate_positions(market)
    assert pos.iloc[0] == -1.0  # short when funding positive
    assert pos.iloc[1] == 1.0   # long when funding negative
    assert pos.iloc[2] == 0.0   # flat at zero funding
