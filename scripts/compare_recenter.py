"""Compare the carry backtest with vs without perp-margin re-centering.

Without re-centering the perp entry reference is fixed for the whole holding
period, so r_spot drifts over weeks and a sustained trend breaches a barrier
that was only sized for a 24h move. With re-centering the entry resets at each
rebalance, matching the horizon the barrier was sized for.

Runs the full (asset, ε, stress) grid both ways and prints a side-by-side
table of liquidation count, annualised return, and liquidation losses.

Usage:
    uv run python scripts/compare_recenter.py
"""
from __future__ import annotations

import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.data.coinbase import CoinbaseClient
from quant_engine.analysis.backtest import BacktestConfig, run_backtest
from quant_engine.analysis.collateral_sizing import theta_F_from_max_leverage

MS_PER_DAY = 86_400_000
HISTORY_DAYS = 400
HL_MAX_LEVERAGE = {"ETH": 50.0, "BTC": 50.0}

ASSETS = ["ETH", "BTC"]
EPS_LEVELS = (0.01, 0.05, 0.10)
STRESS_GRID = (1.0, 1.5, 2.0)


def fetch_with_retry(fn, *args, attempts: int = 4, **kwargs):
    last = None
    for k in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — network flakiness, retry
            last = exc
            time.sleep(1.0 + k)
    raise last


def main() -> None:
    client = HyperliquidClient()          # funding (the venue paying it)
    price_client = CoinbaseClient()       # deep spot price history (full funding cycle)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - HISTORY_DAYS * MS_PER_DAY

    header = (f"{'Asset':>5} {'ε':>5} {'Stress':>7} | "
              f"{'FIXED-ENTRY':>26} | {'RE-CENTERED':>26}")
    sub = (f"{'':>5} {'':>5} {'':>7} | "
           f"{'liq':>4}{'ann':>10}{'loss(bp)':>12} | "
           f"{'liq':>4}{'ann':>10}{'loss(bp)':>12}")

    for asset in ASSETS:
        theta_F = theta_F_from_max_leverage(HL_MAX_LEVERAGE[asset])
        price_df = fetch_with_retry(price_client.fetch_candles, f"{asset}-USD", start_ms, now_ms)
        funding_df = fetch_with_retry(client.fetch_funding_history, asset, start_ms, now_ms)

        print(f"\n{'='*80}")
        print(header)
        print(sub)
        print("-" * 80)

        for eps in EPS_LEVELS:
            for stress in STRESS_GRID:
                cfg_fixed = BacktestConfig(theta_F=theta_F, eps=eps,
                                           stress_mult=stress, recenter_on_rebalance=False)
                cfg_recen = BacktestConfig(theta_F=theta_F, eps=eps,
                                           stress_mult=stress, recenter_on_rebalance=True)
                rf = run_backtest(price_df, funding_df, asset, cfg_fixed)
                rr = run_backtest(price_df, funding_df, asset, cfg_recen)
                flag = "  ←" if rr.ann_return > 0 and rf.ann_return <= 0 else ""
                print(f"{asset:>5} {eps:>5.0%} {stress:>6.1f}× | "
                      f"{rf.liq_count:>4}{rf.ann_return:>+9.2%}{rf.liq_losses*1e4:>12.0f} | "
                      f"{rr.liq_count:>4}{rr.ann_return:>+9.2%}{rr.liq_losses*1e4:>12.0f}{flag}")
        time.sleep(0.3)

    print(f"\n{'='*80}")
    print("  ← = turned net-positive under re-centering")


if __name__ == "__main__":
    main()
