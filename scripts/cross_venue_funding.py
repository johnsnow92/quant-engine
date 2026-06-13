"""Cross-venue funding: Crypto.com (BTCUSD-PERP) vs Binance (via CoinDesk).

Pulls funding from two venues, normalizes to per-hour, reports annualized funding
per venue and the spread, then backtests a price-neutral cross-venue carry.

Usage:
    uv run python scripts/cross_venue_funding.py
"""
from __future__ import annotations

import argparse

from quant_engine.analysis.funding_spread import (
    annualized,
    build_spread,
    cross_venue_carry,
    normalize_per_hour,
)
from quant_engine.data.coindesk import CoinDeskClient
from quant_engine.data.cryptocom import CryptoComClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-venue funding comparison + carry")
    parser.add_argument("--cc-perp", default="BTCUSD-PERP", help="Crypto.com perp")
    parser.add_argument("--bn-instrument", default="BTC-USDT-VANILLA-PERPETUAL", help="CoinDesk-mapped perp")
    parser.add_argument("--bn-market", default="binance")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument(
        "--cc-interval-hours",
        type=float,
        default=1.0,
        help="ASSUMED Crypto.com funding settlement interval — VERIFY this.",
    )
    args = parser.parse_args()

    # Venue A — Crypto.com (interval assumed; funding_hist has no interval field).
    cc = CryptoComClient().fetch_funding(args.cc_perp, count=args.count)
    cc["interval_hours"] = args.cc_interval_hours
    cc = normalize_per_hour(cc)

    # Venue B — Binance funding via CoinDesk (interval comes from the feed).
    bn = CoinDeskClient().fetch_funding(args.bn_market, args.bn_instrument, limit=args.count)
    bn = normalize_per_hour(bn)

    spread = build_spread(cc, bn)
    print(
        f"Crypto.com {args.cc_perp} (assumed {args.cc_interval_hours:g}h funding) vs "
        f"{args.bn_market} {args.bn_instrument} ({bn['interval_hours'].iloc[-1]:g}h funding)"
    )
    print(f"Aligned bars: {len(spread)}")

    print(f"\n{'venue':<16}{'last ann.':>12}{'mean ann.':>12}")
    print("-" * 40)
    print(f"{'Crypto.com':<16}{annualized(cc['funding_per_hour'].iloc[-1]):>12.2%}{annualized(cc['funding_per_hour'].mean()):>12.2%}")
    print(f"{args.bn_market:<16}{annualized(bn['funding_per_hour'].iloc[-1]):>12.2%}{annualized(bn['funding_per_hour'].mean()):>12.2%}")
    print(f"{'spread (B-A)':<16}{annualized(spread['spread_per_hour'].iloc[-1]):>12.2%}{annualized(spread['spread_per_hour'].mean()):>12.2%}")

    carry = cross_venue_carry(spread, rebalance_fee_bps=2.0)
    m = carry.metrics
    print(
        f"\nCross-venue carry (always harvest the spread): tot_ret {m.total_return:.2%}, "
        f"ann {m.annualized_return:.2%}, sharpe {m.sharpe:.2f}, max_dd {m.max_drawdown:.2%}"
    )
    print(
        "\nFunding intervals/conventions differ by venue and the per-hour accrual is "
        "an approximation — verify before trusting magnitudes. Cross-venue basis, "
        "transfer, and borrow frictions are not modeled."
    )


if __name__ == "__main__":
    main()
