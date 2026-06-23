"""Run historical backtest of the spot-perp carry strategy on Hyperliquid data.

Fetches price + funding history for ETH and BTC, then simulates the
risk-constrained carry strategy across a grid of (ε, vol-stress) configs.

Saves results to data/backtest_results.json.

Usage:
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --assets ETH --eps 0.01 0.05
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.data.coinbase import CoinbaseClient
from quant_engine.data.kraken import KrakenDataClient, CARRY_SYMBOLS
from quant_engine.analysis.backtest import (
    BacktestConfig,
    run_backtest_grid,
    results_to_table,
)
from quant_engine.analysis.collateral_sizing import theta_F_from_max_leverage

MS_PER_DAY = 86_400_000
HISTORY_DAYS = 400

HL_MAX_LEVERAGE: dict[str, float] = {"ETH": 50.0, "BTC": 50.0, "SOL": 20.0}

DEFAULT_ASSETS = ["ETH", "BTC"]
DEFAULT_EPS = (0.01, 0.05, 0.10)
DEFAULT_STRESS = (1.0, 1.5, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run carry strategy backtest: Coinbase spot prices (deep history) "
                    "+ Hyperliquid funding")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--eps", nargs="+", type=float, default=list(DEFAULT_EPS))
    parser.add_argument("--stress", nargs="+", type=float, default=list(DEFAULT_STRESS))
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    parser.add_argument("--rebal-hours", type=int, default=24)
    parser.add_argument("--vol-lookback", type=int, default=30,
                        help="Rolling vol lookback in days")
    parser.add_argument("--lgd", type=float, default=1.0,
                        help="Fraction of net margin (α−θ_F) lost on liquidation")
    parser.add_argument("--fee-bps", type=float, default=4.5,
                        help="Taker fee + slippage per leg per trade, in bps (HL taker ≈ 4.5)")
    parser.add_argument("--funding-capture", type=float, default=1.0,
                        help="Execution-friction fraction of POSITIVE funding received "
                             "(HL pays mechanically → 1.0; negative-funding drag is intrinsic)")
    parser.add_argument("--spot-fee-bps", type=float, default=None,
                        help="Spot-leg fee/side in bps (default: --fee-bps). Kraken Pro "
                             "spot taker ≈ 40, maker ≈ 25.")
    parser.add_argument("--perp-fee-bps", type=float, default=None,
                        help="Perp-leg fee/side in bps (default: --fee-bps). Bitnomial US "
                             "perps ≈ 2 ($0.15/contract).")
    parser.add_argument("--theta-f", type=float, default=None,
                        help="Override maintenance-margin fraction θ_F (default: derived "
                             "from venue max leverage). Bitnomial US-perp margin is "
                             "login-gated; pass a conservative value to model it.")
    parser.add_argument("--price-source", choices=["hyperliquid", "coinbase"],
                        default="coinbase",
                        help="Spot/perp price series source. 'coinbase' (default) gives "
                             "deep (years) hourly history vs Hyperliquid's ~200d candle cap, "
                             "covering a full funding cycle. Funding always comes from "
                             "Hyperliquid. Use 'hyperliquid' for venue-native prices.")
    parser.add_argument("--funding-source", choices=["hyperliquid", "kraken"],
                        default="hyperliquid",
                        help="Funding-rate source. 'hyperliquid' (default, hourly) or "
                             "'kraken' (PF_ perps, hourly history; the US-regulated perps "
                             "settle 8h but track the same funding level).")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "backtest_results.json"))
    args = parser.parse_args()

    client = HyperliquidClient()
    price_client = CoinbaseClient() if args.price_source == "coinbase" else client
    kraken_client = KrakenDataClient() if args.funding_source == "kraken" else None
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - args.history_days * MS_PER_DAY

    all_results = []

    for asset in args.assets:
        print(f"\n[{asset}]")
        theta_F = (args.theta_f if args.theta_f is not None
                   else theta_F_from_max_leverage(HL_MAX_LEVERAGE.get(asset, 20.0)))

        print(f"  fetching prices ({args.history_days}d, {args.price_source}) ...",
              end=" ", flush=True)
        try:
            if args.price_source == "coinbase":
                price_df = price_client.fetch_candles(f"{asset}-USD", start_ms, now_ms)
            else:
                price_df = price_client.fetch_candles(asset, "1h", start_ms, now_ms)
            print(f"{len(price_df)} candles")
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            continue

        print(f"  fetching funding ({args.history_days}d, {args.funding_source}) ...",
              end=" ", flush=True)
        try:
            if args.funding_source == "kraken":
                perp_sym = CARRY_SYMBOLS[asset]["perp"]
                funding_df = kraken_client.fetch_funding(perp_sym)
                funding_df = funding_df[
                    (funding_df["ts"] >= start_ms) & (funding_df["ts"] <= now_ms)
                ].reset_index(drop=True)
            else:
                funding_df = client.fetch_funding_history(asset, start_ms, now_ms)
            print(f"{len(funding_df)} records")
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            continue

        base_cfg = BacktestConfig(
            lgd=args.lgd,
            rebal_hours=args.rebal_hours,
            vol_lookback_days=args.vol_lookback,
            carry_lookback_days=args.vol_lookback,
            theta_F=theta_F,
            fee_bps_per_side=args.fee_bps,
            spot_fee_bps_per_side=args.spot_fee_bps,
            perp_fee_bps_per_side=args.perp_fee_bps,
            funding_capture=args.funding_capture,
        )

        results = run_backtest_grid(
            price_df, funding_df, asset,
            eps_levels=tuple(args.eps),
            stress_mults=tuple(args.stress),
            base_config=base_cfg,
        )
        all_results.extend(results)

        for r in results:
            print(
                f"  ε={r.config.eps:.0%} stress={r.config.stress_mult:.1f}×  "
                f"ann={r.ann_return:.2%}  Sharpe={r.sharpe:.2f}  liq={r.liq_count}  "
                f"gross={r.gross_carry*1e4:.0f}bp  drag={r.funding_drag*1e4:.0f}bp  "
                f"fees={r.fees*1e4:.0f}bp  net={r.net_pnl*1e4:.0f}bp"
            )

        time.sleep(0.3)

    if not all_results:
        print("\nNo results.")
        return

    print("\n" + "=" * 90)
    print(results_to_table(all_results))
    print("=" * 90)

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assets": args.assets,
        "price_source": args.price_source,
        "funding_source": args.funding_source,
        "history_days": args.history_days,
        "eps_levels": args.eps,
        "stress_mults": args.stress,
        "results": [
            {
                "asset": r.asset,
                "eps": r.config.eps,
                "stress_mult": r.config.stress_mult,
                "lgd": r.config.lgd,
                "theta_F": r.config.theta_F,
                "fee_bps_per_side": r.config.fee_bps_per_side,
                "funding_capture": r.config.funding_capture,
                "n_hours": r.n_hours,
                "n_active_hours": r.n_active_hours,
                "liq_count": r.liq_count,
                "total_return": r.total_return,
                "ann_return": r.ann_return,
                "ann_vol": r.ann_vol,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "gross_carry": r.gross_carry,
                "funding_drag": r.funding_drag,
                "fees": r.fees,
                "liq_losses": r.liq_losses,
                "net_pnl": r.net_pnl,
                "equity_daily": r.equity_series[::24].tolist(),
                "alpha_daily": r.alpha_series[::24].tolist(),
            }
            for r in all_results
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
