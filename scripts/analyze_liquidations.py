"""Path analysis of backtest liquidation events.

Answers: why does only BTC at ε=1% / 2× vol stress survive with zero
liquidations, while every other config gets liquidated?

For each (asset, ε, stress) config it runs the real backtest with an event
log, then prints every liquidation event with the price move that breached
the barrier, the α / r_liq held at the time, and the σ_h that sized α.

It also characterises each asset's price path: the distribution of forward
24h upward returns (the tail that liquidates a short perp) and the single
largest upward excursion, so we can see whether each config's barrier sat
above or below the moves that actually happened.

Usage:
    uv run python scripts/analyze_liquidations.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.data.coinbase import CoinbaseClient
from quant_engine.analysis.backtest import (
    BacktestConfig,
    run_backtest,
    merge_price_funding,
)
from quant_engine.analysis.collateral_sizing import (
    theta_F_from_max_leverage,
    liq_barrier,
    alpha_risk_constrained,
)

MS_PER_DAY = 86_400_000
HISTORY_DAYS = 400
HL_MAX_LEVERAGE = {"ETH": 50.0, "BTC": 50.0}

ASSETS = ["ETH", "BTC"]
EPS = 0.01
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


def fmt_ts(ms: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000.0))


def characterise_path(merged, warm_up: int, horizon: int = 24) -> dict:
    """Forward-`horizon`h max upward return from each active hour."""
    prices = merged["close"].to_numpy(dtype=float)
    n = len(prices)
    fwd_max_up = []
    for i in range(warm_up, n - horizon):
        window = prices[i + 1: i + 1 + horizon]
        fwd_max_up.append(float(window.max() / prices[i] - 1.0))
    arr = np.array(fwd_max_up) if fwd_max_up else np.array([0.0])
    # Also: largest single upward move over the whole active window
    active = prices[warm_up:]
    log_rets = np.diff(np.log(active))
    return {
        "n_active": n - warm_up,
        "fwd24h_p50": float(np.percentile(arr, 50)),
        "fwd24h_p95": float(np.percentile(arr, 95)),
        "fwd24h_p99": float(np.percentile(arr, 99)),
        "fwd24h_max": float(arr.max()),
        "hourly_vol": float(log_rets.std(ddof=1)),
        "ann_vol": float(log_rets.std(ddof=1) * np.sqrt(8760.0)),
    }


def main() -> None:
    client = HyperliquidClient()          # funding (the venue paying it)
    price_client = CoinbaseClient()       # deep spot price history (full funding cycle)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - HISTORY_DAYS * MS_PER_DAY

    for asset in ASSETS:
        theta_F = theta_F_from_max_leverage(HL_MAX_LEVERAGE[asset])
        print(f"\n{'='*78}\n{asset}  (θ_F={theta_F:.4f}, L_max={HL_MAX_LEVERAGE[asset]:.0f}×)\n{'='*78}")

        price_df = fetch_with_retry(price_client.fetch_candles, f"{asset}-USD", start_ms, now_ms)
        funding_df = fetch_with_retry(client.fetch_funding_history, asset, start_ms, now_ms)
        merged = merge_price_funding(price_df, funding_df)

        base_cfg = BacktestConfig(theta_F=theta_F, eps=EPS)
        warm_up = base_cfg.vol_lookback_hours
        path = characterise_path(merged, warm_up)

        print(f"  merged hours: {len(merged)}  active: {path['n_active']} "
              f"({path['n_active']/24:.0f}d)")
        print(f"  realized hourly σ: {path['hourly_vol']:.4f}  "
              f"(annualised {path['ann_vol']:.1%})")
        print(f"  forward-24h max upward move  "
              f"p50={path['fwd24h_p50']:+.2%}  p95={path['fwd24h_p95']:+.2%}  "
              f"p99={path['fwd24h_p99']:+.2%}  MAX={path['fwd24h_max']:+.2%}")

        for stress in STRESS_GRID:
            cfg = BacktestConfig(theta_F=theta_F, eps=EPS, stress_mult=stress)
            events: list = []
            result = run_backtest(price_df, funding_df, asset, cfg, event_log=events)

            # α/r_liq the config typically holds (median over active hours)
            active_alpha = result.alpha_series[warm_up:]
            med_alpha = float(np.median(active_alpha[active_alpha > 0])) if np.any(active_alpha > 0) else 0.0
            med_barrier = liq_barrier(med_alpha, theta_F) - 1.0 if med_alpha > 0 else float("nan")

            print(f"\n  ── ε={EPS:.0%}  stress={stress:.1f}×  "
                  f"liq={result.liq_count}  ann={result.ann_return:+.2%}  "
                  f"carry={result.gross_carry*1e4:.0f}bp  loss={result.liq_losses*1e4:.0f}bp")
            print(f"     median α≈{med_alpha:.3f}  →  median barrier ≈ +{med_barrier:.2%} spot move "
                  f"(vs realized 24h-max {path['fwd24h_max']:+.2%})")

            if not events:
                print("     NO LIQUIDATIONS — barrier held above every move in the path")
                continue

            print(f"     {'liq date':<17}{'held':>7}{'move':>9}{'barrier':>9}"
                  f"{'α':>7}{'σ_h·s':>9}{'loss(bp)':>10}")
            for e in events:
                held_d = e["hours_held"] / 24.0
                print(f"     {fmt_ts(e['timestamp_ms']):<17}"
                      f"{held_d:>5.0f}d "
                      f"{e['pct_move']:>+8.2%} "
                      f"{e['barrier_move_pct']:>+8.2%} "
                      f"{e['alpha']:>6.3f} "
                      f"{e['sigma_h_stressed']:>8.4f} "
                      f"{e['loss']*1e4:>9.1f}")
                print(f"       └ entered {fmt_ts(e['entry_timestamp_ms'])} "
                      f"@ {e['entry_price']:.1f} → liq @ {e['trigger_price']:.1f}")

        time.sleep(0.3)


if __name__ == "__main__":
    main()
