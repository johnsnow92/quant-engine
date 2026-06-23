"""Calibrate Ornstein-Uhlenbeck funding-rate parameters from live Hyperliquid data.

Fetches hourly funding history for ETH, BTC, SOL, fits the OU model
(kappa, theta, sigma_f) per asset for 90/180/360-day windows, and saves
the results to data/ou_params.json.

Paper reference: arXiv:2605.06405 (Table 1, Section 6.2).
Expected half-lives: ETH ~5.6h, BTC ~4.1h, SOL ~2.3h.

Usage:
    uv run python scripts/calibrate_ou_funding.py
    uv run python scripts/calibrate_ou_funding.py --assets BTC ETH --window 180
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
from quant_engine.analysis.ou_calibration import calibrate_multi_window, params_to_dict

DEFAULT_ASSETS = ["ETH", "BTC", "SOL"]
DEFAULT_WINDOWS = (90, 180, 360)
HISTORY_DAYS = 400           # fetch enough for the 360-day window
MS_PER_DAY = 86_400_000

# Paper's Table 1 benchmarks for sanity check
PAPER_HALF_LIVES = {"ETH": 5.560, "BTC": 4.071, "SOL": 2.310}


def fetch_history(client: HyperliquidClient, coin: str, days: int) -> "pd.DataFrame":
    import time as _time
    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * MS_PER_DAY
    print(f"  fetching {coin} ({days}d) ...", end=" ", flush=True)
    df = client.fetch_funding_history(coin, start_ms, now_ms)
    print(f"{len(df)} records")
    return df


def print_table(all_params: list) -> None:
    header = f"{'Asset':>6} {'Days':>5} {'N':>6} {'kappa':>10} {'theta':>12} {'sigma_f':>12} {'t½ (h)':>8} {'jump%/h':>9} {'paper t½':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for p in all_params:
        paper_hl = PAPER_HALF_LIVES.get(p.asset, float("nan"))
        print(
            f"{p.asset:>6} {p.window_days:>5} {p.n_obs:>6} "
            f"{p.kappa:>10.5f} {p.theta:>12.6e} {p.sigma_f:>12.6e} "
            f"{p.half_life_hours:>8.2f} {p.jump_prob_per_hour:>9.2%} {paper_hl:>9.3f}"
        )
    print("=" * len(header))
    print("  kappa in h⁻¹ | theta/sigma_f in fractional per-hour rate | t½ = log(2)/kappa")


def print_hjb_grid(all_params: list, window: int = 180) -> None:
    selected = [p for p in all_params if p.window_days == window]
    if not selected:
        return
    print(f"\nHJB grid bounds (±4σ stationary, {window}d window):")
    for p in selected:
        print(f"  {p.asset}: f_grid = [{p.f_grid_min:.4e}, {p.f_grid_max:.4e}]  "
              f"(theta={p.theta:.4e}, sigma_stat={p.sigma_stationary:.4e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate OU funding-rate params from Hyperliquid")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "ou_params.json"))
    args = parser.parse_args()

    client = HyperliquidClient()
    all_params = []

    for asset in args.assets:
        print(f"\n[{asset}]")
        try:
            df = fetch_history(client, asset, args.history_days)
        except Exception as exc:
            print(f"  ERROR fetching {asset}: {exc}")
            continue

        params_list = calibrate_multi_window(
            df, asset, tuple(args.windows)
        )
        for p in params_list:
            all_params.append(p)
            # brief per-window line
            paper_hl = PAPER_HALF_LIVES.get(asset, float("nan"))
            delta = p.half_life_hours - paper_hl
            flag = "  ✓" if abs(delta) < 2.0 else f"  Δ={delta:+.2f}h vs paper"
            print(f"  {p.window_days:>3}d: kappa={p.kappa:.5f}  theta={p.theta:.4e}  "
                  f"sigma_f={p.sigma_f:.4e}  t½={p.half_life_hours:.2f}h{flag}")

        time.sleep(0.3)   # gentle rate-limiting between assets

    if not all_params:
        print("\nNo results — check network / asset names.")
        return

    print_table(all_params)
    print_hjb_grid(all_params, window=180)

    # save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assets": args.assets,
        "windows_days": args.windows,
        "params": [params_to_dict(p) for p in all_params],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
