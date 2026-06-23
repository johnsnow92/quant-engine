"""Real-money funding-carry validation on Kraken (build #2).

Pulls REAL Kraken data — spot OHLC + perp candles + hourly perpetual funding —
and backtests the market-neutral funding carry (short perp + long spot when
funding is positive, inverse when negative) with real funding and realistic
fees. Reports per-asset Sharpe / max-DD / net return and ``price_beta`` (~0
confirms the spot hedge neutralizes price, so PnL is funding, not direction).

Optionally prints a cross-venue funding snapshot from Coinalyze (set
COINALYZE_API_KEY) so you can see whether Kraken's funding is competitive before
committing the carry to Kraken execution.

No credentials or capital required — all data is public. This is the
build-and-validate pass before any live cutover (which needs Kraken Futures API
creds + a funded account).

Usage:
    uv run python scripts/run_kraken_carry.py
    uv run python scripts/run_kraken_carry.py --assets BTC ETH SOL --threshold 5e-6
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_engine.backtest.carry import run_carry_backtest
from quant_engine.backtest.metrics import price_beta
from quant_engine.data.coinalyze import CoinalyzeClient, CoinalyzeError
from quant_engine.data.kraken import CARRY_SYMBOLS, KrakenDataClient
from quant_engine.strategies.carry_signal import FundingCarry

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
HOURS_PER_YEAR = 8760


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Kraken funding-carry validation")
    parser.add_argument("--assets", nargs="+", default=list(CARRY_SYMBOLS))
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="funding magnitude (fraction/hr) to enter; 0 = always follow sign")
    parser.add_argument("--perp-fee-bps", type=float, default=5.0)
    parser.add_argument("--spot-fee-bps", type=float, default=10.0)
    parser.add_argument("--scan-venues", action="store_true",
                        help="print Coinalyze cross-venue funding snapshot (needs COINALYZE_API_KEY)")
    args = parser.parse_args()

    client = KrakenDataClient(cache_dir=CACHE)
    signal = FundingCarry(threshold=args.threshold)

    header = (f"{'asset':<7}{'bars':>6}{'tot_ret':>10}{'ann_ret':>10}"
              f"{'sharpe':>9}{'max_dd':>10}{'price_beta':>12}{'trades':>8}{'perp':>7}")
    print(header)
    print("-" * len(header))
    for asset in args.assets:
        try:
            market = client.load_carry_market(asset)
        except Exception as exc:  # noqa: BLE001 — surface data issues per-asset, keep going
            print(f"{asset:<7}  data error: {exc}")
            continue
        carry_sign = signal.generate_carry_positions(market)
        res = run_carry_backtest(
            market, carry_sign,
            perp_fee_bps=args.perp_fee_bps, spot_fee_bps=args.spot_fee_bps,
            periods_per_year=HOURS_PER_YEAR,
        )
        beta = price_beta(res.returns, market["spot_close"].pct_change().fillna(0.0))
        m = res.metrics
        perp_tag = "real" if market.attrs.get("perp_real") else "=spot"
        print(f"{asset:<7}{len(market):>6}{m.total_return:>10.2%}{m.annualized_return:>10.2%}"
              f"{m.sharpe:>9.2f}{m.max_drawdown:>10.2%}{beta:>12.4f}{m.num_trades:>8d}{perp_tag:>7}")

    print("\nprice_beta ~0 confirms the hedge: carry PnL is funding, not price direction.")
    print("Window is ~30 days of hourly bars (Kraken spot OHLC cap); extend via "
          "CoinDesk MCP / paginated history for a longer walk-forward.")

    if args.scan_venues:
        print("\n--- Cross-venue funding (Coinalyze), richest-to-short first ---")
        try:
            cv = CoinalyzeClient()
            for base in args.assets:
                snap = cv.cross_venue_funding(base)
                if snap.empty:
                    print(f"{base}: no venues returned")
                    continue
                top = snap.head(3).to_dict("records")
                print(f"{base}: " + " | ".join(
                    f"{r['symbol']} {r['funding_rate']:+.6f}" for r in top))
        except CoinalyzeError as exc:
            print(f"(skipped) {exc}")


if __name__ == "__main__":
    main()
