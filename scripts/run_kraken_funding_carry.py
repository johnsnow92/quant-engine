"""Year-long funding-carry economics on REAL Kraken funding (the decisive test).

Kraken's public funding endpoint returns ~1 year of HOURLY funding per perp
(no 720-bar cap), so this judges the carry across multiple regimes. The carry
PnL is ``held * funding - fees`` and fees depend only on the signal's turnover,
not price — so this isolates the necessary condition: does the funding actually
collected beat the fees of harvesting it, over a full year, with a LOW-TURNOVER
signal and REALISTIC Kraken fee tiers?

The perp/spot basis is approximated to ~0 here (perp_close == spot_close), which
makes drawdown OPTIMISTIC — real basis blowouts add DD. So a positive result is
NECESSARY-not-sufficient: the follow-on is a basis-risk test on deep spot+perp
price history. A non-positive result is decisive on its own.

Fee scenarios (per side, bps) reflect Kraken's real tiers:
  maker_hi_vol : perp 2  + spot 10   (maker, high-volume tier)
  maker_lo_vol : perp 2  + spot 25   (maker, entry tier)
  taker_lo_vol : perp 5  + spot 40   (taker, entry tier)

Usage:
    uv run python scripts/run_kraken_funding_carry.py
    uv run python scripts/run_kraken_funding_carry.py --smooth 48 --enter 2e-5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_engine.backtest.carry import run_carry_backtest
from quant_engine.data.kraken import CARRY_SYMBOLS, KrakenDataClient
from quant_engine.strategies.carry_signal import SmoothedFundingCarry

HOURS_PER_YEAR = 8760
FEE_SCENARIOS = {
    "maker_hi_vol": (2.0, 10.0),
    "maker_lo_vol": (2.0, 25.0),
    "taker_lo_vol": (5.0, 40.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Year-long Kraken funding-carry economics")
    parser.add_argument("--assets", nargs="+", default=list(CARRY_SYMBOLS))
    parser.add_argument("--smooth", type=int, default=24, help="funding smoothing window (hours)")
    parser.add_argument("--enter", type=float, default=1e-5, help="smoothed funding/hr to open")
    parser.add_argument("--exit", dest="exit_t", type=float, default=0.0, help="smoothed funding/hr to flatten")
    args = parser.parse_args()

    client = KrakenDataClient()
    signal = SmoothedFundingCarry(args.smooth, args.enter, args.exit_t)

    for asset in args.assets:
        sym = CARRY_SYMBOLS[asset]["perp"]
        funding = client.fetch_funding(sym)
        # Basis approximated to 0: equal, flat price legs isolate funding-vs-fees.
        market = funding.copy()
        market["perp_close"] = 1.0
        market["spot_close"] = 1.0
        carry_sign = signal.generate_carry_positions(market)
        span_days = (funding["ts"].iloc[-1] - funding["ts"].iloc[0]) / 86_400_000.0
        gross_funding = float((carry_sign.shift(1).fillna(0.0) * market["funding_rate"]).sum())

        print(f"\n=== {asset}  ({len(funding)} hourly bars, ~{span_days:.0f} days, "
              f"perp {sym}) ===")
        print(f"  gross funding captured (pre-fee): {gross_funding:>8.2%}   "
              f"exposure {float((carry_sign != 0).mean()):.0%}")
        hdr = f"  {'fee scenario':<14}{'net_ret':>10}{'ann_ret':>10}{'sharpe':>9}{'max_dd':>10}{'trades':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for name, (perp_bps, spot_bps) in FEE_SCENARIOS.items():
            res = run_carry_backtest(
                market, carry_sign, perp_fee_bps=perp_bps, spot_fee_bps=spot_bps,
                periods_per_year=HOURS_PER_YEAR,
            )
            m = res.metrics
            print(f"  {name:<14}{m.total_return:>10.2%}{m.annualized_return:>10.2%}"
                  f"{m.sharpe:>9.2f}{m.max_drawdown:>10.2%}{m.num_trades:>8d}")

    print("\nNOTE: basis approximated to ~0 (perp==spot), so max_dd is OPTIMISTIC. "
          "Positive net here is necessary-not-sufficient; next test is basis risk "
          "on deep spot+perp price history.")


if __name__ == "__main__":
    main()
