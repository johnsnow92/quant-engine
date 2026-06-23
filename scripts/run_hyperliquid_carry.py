"""Hyperliquid funding-carry validation — net-of-fee verdict on the hunt's winners.

The cross-venue hunt flagged persistently rich Hyperliquid funding (13-45%/yr on
several alts). This pulls each candidate's REAL hourly funding from Hyperliquid's
native API over a multi-month window and runs the low-turnover carry with
realistic fees to answer: does it actually net out positive after the perp fee
AND the spot-hedge leg?

Basis approximated to ~0 (isolates funding-vs-fees), so max_dd is OPTIMISTIC —
the real risks are spot-hedge availability/cost for thin alts and basis blowouts.
A positive net here is necessary-not-sufficient; it says the funding clears fees,
which is exactly where the Kraken carry failed.

Fee scenarios (per side, bps):
  hl_maker+major  : perp 1.5 + spot 10   (HL maker, liquid spot hedge, maker)
  hl_taker+alt    : perp 4.5 + spot 35   (HL taker, thin-alt spot hedge, taker)

Usage:
    uv run python scripts/run_hyperliquid_carry.py
    uv run python scripts/run_hyperliquid_carry.py --coins XMR HEMI ZRO --days 180
"""
from __future__ import annotations

import argparse
import time

from quant_engine.backtest.carry import run_carry_backtest
from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.strategies.carry_signal import SmoothedFundingCarry

HOURS_PER_YEAR = 8760
DEFAULT_COINS = ["XMR", "HEMI", "MANTA", "ZRO", "GRASS", "DYDX", "HYPE", "ENS", "RSR", "TRB"]
FEE_SCENARIOS = {"hl_maker+major": (1.5, 10.0), "hl_taker+alt": (4.5, 35.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperliquid funding-carry validation")
    parser.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    parser.add_argument("--days", type=int, default=150)
    parser.add_argument("--smooth", type=int, default=24)
    parser.add_argument("--enter", type=float, default=5e-6)
    args = parser.parse_args()

    client = HyperliquidClient()
    signal = SmoothedFundingCarry(args.smooth, args.enter, 0.0)
    now = int(time.time() * 1000)
    start = now - args.days * 86_400_000

    hdr = (f"{'coin':<8}{'bars':>6}{'days':>6}{'gross_ann':>11}"
           f"{'  scenario':<16}{'net_ann':>9}{'sharpe':>8}{'max_dd':>9}{'trades':>8}")
    print(hdr)
    print("-" * len(hdr))
    for coin in args.coins:
        try:
            funding = client.fetch_funding_history(coin, start, now)
        except Exception as exc:  # noqa: BLE001
            print(f"{coin:<8}  data error: {exc}")
            continue
        market = funding.copy()
        market["perp_close"] = 1.0
        market["spot_close"] = 1.0
        carry_sign = signal.generate_carry_positions(market)
        held = carry_sign.shift(1).fillna(0.0)
        span_days = (funding["ts"].iloc[-1] - funding["ts"].iloc[0]) / 86_400_000.0
        gross = float((held * market["funding_rate"]).sum())
        gross_ann = gross / max(span_days, 1) * 365

        first = True
        for name, (perp_bps, spot_bps) in FEE_SCENARIOS.items():
            res = run_carry_backtest(market, carry_sign, perp_fee_bps=perp_bps,
                                     spot_fee_bps=spot_bps, periods_per_year=HOURS_PER_YEAR)
            m = res.metrics
            net_ann = m.total_return / max(span_days, 1) * 365
            lead = (f"{coin:<8}{len(funding):>6}{span_days:>6.0f}{gross_ann:>10.1%}"
                    if first else " " * 31)
            print(f"{lead}{'  ' + name:<16}{net_ann:>9.1%}{m.sharpe:>8.2f}"
                  f"{m.max_drawdown:>9.2%}{m.num_trades:>8d}")
            first = False

    print("\ngross_ann/net_ann annualized from the realized window. Basis ~0 => max_dd "
          "OPTIMISTIC. Positive net = funding clears fees (where Kraken failed); the "
          "open risks are spot-hedge availability/cost for thin alts + basis/longer-horizon.")


if __name__ == "__main__":
    main()
