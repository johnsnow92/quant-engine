"""Fetch a crypto perp, run the bootstrap strategies, print a metrics table.

Usage:
    uv run python scripts/run_backtest.py --instrument BTCUSD-PERP --timeframe 1h --count 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_engine.backtest.engine import run_backtest
from quant_engine.data.cryptocom import CryptoComClient
from quant_engine.strategies.funding_tilt import FundingTilt
from quant_engine.strategies.ma_crossover import MACrossover

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crypto perp backtests")
    parser.add_argument("--instrument", default="BTCUSD-PERP")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    client = CryptoComClient(cache_dir=CACHE)
    market = client.load_market(
        args.instrument, args.timeframe, args.count, use_cache=not args.no_cache
    )
    print(
        f"Loaded {len(market)} bars for {args.instrument} {args.timeframe} "
        f"({market['time'].iloc[0]} -> {market['time'].iloc[-1]})"
    )

    strategies = [MACrossover(), FundingTilt()]
    header = f"{'strategy':<16}{'tot_ret':>10}{'ann_ret':>11}{'sharpe':>9}{'max_dd':>10}{'trades':>8}{'expo':>7}"
    print("\n" + header)
    print("-" * len(header))
    for strat in strategies:
        positions = strat.generate_positions(market)
        result = run_backtest(market, positions, fee_bps=args.fee_bps)
        m = result.metrics
        print(
            f"{strat.name:<16}{m.total_return:>10.2%}{m.annualized_return:>11.2%}"
            f"{m.sharpe:>9.2f}{m.max_drawdown:>10.2%}{m.num_trades:>8d}{m.exposure:>7.0%}"
        )

    print(
        "\nNote: funding-rate convention varies by venue — verify Crypto.com's "
        "funding unit before trusting carry PnL. This run proves the pipeline, "
        "not a tradable edge."
    )


if __name__ == "__main__":
    main()
