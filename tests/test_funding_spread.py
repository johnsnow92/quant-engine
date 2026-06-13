import pandas as pd
import pytest

from quant_engine.analysis.funding_spread import (
    annualized,
    build_spread,
    cross_venue_carry,
    normalize_per_hour,
)


def test_normalize_per_hour_divides_by_interval():
    df = pd.DataFrame({"ts": [1, 2], "funding_rate": [0.0008, 0.0008], "interval_hours": [8.0, 8.0]})
    out = normalize_per_hour(df)
    assert out["funding_per_hour"].iloc[0] == pytest.approx(0.0001)


def test_annualized():
    assert annualized(0.0001) == pytest.approx(0.876)  # 1e-4 * 8760


def test_build_spread_inner_joins_on_ts():
    a = pd.DataFrame({"ts": [1, 2, 3], "funding_per_hour": [1e-6, 1e-6, 1e-6]})
    b = pd.DataFrame({"ts": [2, 3, 4], "funding_per_hour": [3e-6, 2e-6, 5e-6]})
    spread = build_spread(a, b)
    assert list(spread["ts"]) == [2, 3]
    assert spread["spread_per_hour"].iloc[0] == pytest.approx(2e-6)  # 3e-6 - 1e-6


def test_cross_venue_carry_collects_positive_spread():
    spread = pd.DataFrame({"ts": [1, 2, 3, 4], "spread_per_hour": [1e-4] * 4})
    res = cross_venue_carry(spread, rebalance_fee_bps=0.0)
    assert res.returns.iloc[1:].to_numpy() == pytest.approx([1e-4, 1e-4, 1e-4])


def test_cross_venue_carry_harvests_negative_spread_too():
    # A negative spread is collected by taking the opposite side.
    spread = pd.DataFrame({"ts": [1, 2, 3], "spread_per_hour": [-1e-4, -1e-4, -1e-4]})
    res = cross_venue_carry(spread, rebalance_fee_bps=0.0)
    assert (res.returns.iloc[1:] > 0).all()
