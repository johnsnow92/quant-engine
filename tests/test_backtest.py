import numpy as np
import pandas as pd

from quant_engine.backtest.engine import run_backtest


def make_market(prices, funding=0.0):
    n = len(prices)
    return pd.DataFrame({"close": prices, "funding_rate": [funding] * n})


def test_long_position_on_rising_prices_profits():
    market = make_market([100, 110, 121, 133.1])  # +10% per bar
    positions = pd.Series([1.0] * 4)
    result = run_backtest(market, positions, fee_bps=0.0, include_funding=False)
    assert result.equity.iloc[-1] > 1.0
    assert result.metrics.total_return > 0


def test_no_lookahead_first_bar_flat():
    market = make_market([100, 110, 121])
    positions = pd.Series([1.0, 1.0, 1.0])
    result = run_backtest(market, positions, fee_bps=0.0, include_funding=False)
    assert result.returns.iloc[0] == 0.0  # position is shifted, first bar earns nothing


def test_funding_reduces_long_return_on_flat_price():
    market = make_market([100, 100, 100], funding=0.001)
    positions = pd.Series([1.0, 1.0, 1.0])
    result = run_backtest(market, positions, fee_bps=0.0, include_funding=True)
    assert result.equity.iloc[-1] < 1.0  # long pays positive funding


def test_short_harvests_positive_funding():
    market = make_market([100, 100, 100], funding=0.001)
    positions = pd.Series([-1.0, -1.0, -1.0])
    result = run_backtest(market, positions, fee_bps=0.0, include_funding=True)
    assert result.equity.iloc[-1] > 1.0  # short collects positive funding


def test_metrics_are_finite():
    rng = np.random.default_rng(42)
    prices = (100 * np.cumprod(1 + rng.normal(0, 0.01, 200))).tolist()
    market = make_market(prices)
    positions = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=200))
    result = run_backtest(market, positions, fee_bps=5.0)
    m = result.metrics
    assert np.isfinite(m.sharpe)
    assert np.isfinite(m.max_drawdown)
    assert m.num_trades >= 0
    assert 0.0 <= m.exposure <= 1.0
