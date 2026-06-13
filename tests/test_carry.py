import numpy as np
import pandas as pd

from quant_engine.backtest.carry import run_carry_backtest
from quant_engine.backtest.metrics import price_beta
from quant_engine.strategies.carry_signal import FundingCarry


def make_carry_market(perp, spot, funding):
    return pd.DataFrame({"perp_close": perp, "spot_close": spot, "funding_rate": funding})


def test_perfect_hedge_isolates_funding():
    # Perp tracks spot exactly. PnL must equal funding each bar regardless of the
    # (wild) price path — the price legs cancel.
    prices = [100, 130, 90, 145, 70]
    market = make_carry_market(prices, prices, [0.001] * 5)
    sign = pd.Series([1.0] * 5)
    result = run_carry_backtest(market, sign, perp_fee_bps=0.0, spot_fee_bps=0.0)
    np.testing.assert_allclose(result.returns.iloc[1:].to_numpy(), [0.001] * 4, atol=1e-12)


def test_carry_is_price_neutral():
    rng = np.random.default_rng(7)
    prices = (100 * np.cumprod(1 + rng.normal(0, 0.02, 200))).tolist()
    market = make_carry_market(prices, prices, [0.0005] * 200)
    sign = pd.Series([1.0] * 200)
    result = run_carry_backtest(market, sign, perp_fee_bps=0.0, spot_fee_bps=0.0)
    perp_ret = pd.Series(prices).pct_change().fillna(0.0)
    # Over the held window (drop the flat entry bar) the carry return is constant
    # funding, so price sensitivity is exactly zero.
    assert abs(price_beta(result.returns.iloc[1:], perp_ret.iloc[1:])) < 1e-12


def test_fees_charged_on_both_legs():
    market = make_carry_market([100, 100, 100], [100, 100, 100], [0.001, -0.001, 0.001])
    sign = pd.Series([1.0, -1.0, 1.0])  # held = [0, 1, -1], turnover = [0, 1, 2]
    result = run_carry_backtest(market, sign, perp_fee_bps=5.0, spot_fee_bps=10.0)
    nofee = run_carry_backtest(market, sign, perp_fee_bps=0.0, spot_fee_bps=0.0)
    # bar 1: turnover 1 -> fee (5 + 10) bps = 0.0015 charged on top of gross
    assert result.returns.iloc[1] == nofee.returns.iloc[1] - 0.0015


def test_no_lookahead_first_bar_flat():
    market = make_carry_market([100, 110, 121], [100, 110, 121], [0.001, 0.001, 0.001])
    sign = pd.Series([1.0, 1.0, 1.0])
    result = run_carry_backtest(market, sign, perp_fee_bps=0.0, spot_fee_bps=0.0)
    assert result.returns.iloc[0] == 0.0


def test_funding_carry_signal_direction():
    market = pd.DataFrame({"funding_rate": [0.001, -0.001, 0.0]})
    sign = FundingCarry().generate_carry_positions(market)
    assert sign.iloc[0] == 1.0   # positive funding -> short perp + long spot
    assert sign.iloc[1] == -1.0  # negative funding -> long perp + short spot
    assert sign.iloc[2] == 0.0
