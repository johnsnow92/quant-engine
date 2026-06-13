"""Compare directional funding tilt vs market-neutral funding carry on live data.

The point of this script is the ``price_beta`` column: the directional tilt
carries real price exposure, the hedged carry should be ~0.

Usage:
    uv run python scripts/run_carry.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_engine.backtest.carry import run_carry_backtest
from quant_engine.backtest.engine import run_backtest
from quant_engine.backtest.metrics import price_beta
from quant_engine.data.cryptocom import CryptoComClient
from quant_engine.strategies.carry_signal import FundingCarry
from quant_engine.strategies.funding_tilt import FundingTilt

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Directional tilt vs market-neutral carry")
    parser.add_argument("--perp", default="BTCUSD-PERP")
    parser.add_argument("--spot", default="BTC_USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    client = CryptoComClient(cache_dir=CACHE)
    market = client.load_carry_market(
        args.perp, args.spot, args.timeframe, args.count, use_cache=not args.no_cache
    )
    print(
        f"Loaded {len(market)} aligned bars {args.perp}/{args.spot} "
        f"({market['time'].iloc[0]} -> {market['time'].iloc[-1]})"
    )

    perp_ret = market["perp_close"].pct_change().fillna(0.0)

    # Directional: funding tilt on the perp only — still fully price-exposed.
    tilt_market = market.rename(columns={"perp_close": "close"})
    tilt = run_backtest(tilt_market, FundingTilt().generate_positions(tilt_market), fee_bps=5.0)
    tilt_beta = price_beta(tilt.returns, perp_ret)

    # Market-neutral: short perp + long spot (price leg hedged).
    carry_sign = FundingCarry().generate_carry_positions(market)
    carry = run_carry_backtest(market, carry_sign, perp_fee_bps=5.0, spot_fee_bps=10.0)
    carry_beta = price_beta(carry.returns, perp_ret)

    header = f"{'strategy':<22}{'tot_ret':>10}{'sharpe':>9}{'max_dd':>10}{'price_beta':>12}{'trades':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, res, beta in [
        ("directional tilt", tilt, tilt_beta),
        ("market-neutral carry", carry, carry_beta),
    ]:
        m = res.metrics
        print(
            f"{name:<22}{m.total_return:>10.2%}{m.sharpe:>9.2f}"
            f"{m.max_drawdown:>10.2%}{beta:>12.4f}{m.num_trades:>8d}"
        )

    print(
        "\nprice_beta ~0 means the spot hedge neutralized price — the carry's PnL "
        "is funding, not direction. Verify each venue's funding interval/units "
        "before trusting magnitudes; this proves the hedge, not an edge."
    )


if __name__ == "__main__":
    main()
