"""Compute HJB-motivated collateral sizing from live Hyperliquid data.

Fetches hourly candles to compute realized vol, loads OU params from
data/ou_params.json for the carry estimate, then runs risk-constrained
and economic collateral optimization per asset across stress scenarios.

Saves results to data/collateral_sizing.json.

Usage:
    uv run python scripts/compute_collateral_sizing.py
    uv run python scripts/compute_collateral_sizing.py --assets BTC ETH --horizon 48
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.analysis.collateral_sizing import (
    theta_F_from_max_leverage,
    size_collateral,
    results_to_table,
)
from quant_engine.analysis.ou_calibration import params_to_dict

MS_PER_DAY = 86_400_000
VOL_LOOKBACK_DAYS = 90

# Hyperliquid max leverage per asset (→ maintenance margin θ_F = 1/(2·L_max))
HL_MAX_LEVERAGE: dict[str, float] = {
    "ETH": 50.0,
    "BTC": 50.0,
    "SOL": 20.0,
}

DEFAULT_ASSETS = ["ETH", "BTC", "SOL"]
EPS_LEVELS = (0.01, 0.05, 0.10)
STRESS_MULTS = (1.0, 1.5, 2.0)


def load_ou_params(path: Path) -> dict[str, dict]:
    """Load ou_params.json; return {asset -> best-window OUParams dict}."""
    with open(path) as f:
        payload = json.load(f)
    by_asset: dict[str, dict] = {}
    for entry in payload["params"]:
        asset = entry["asset"]
        window = entry["window_days"]
        prev = by_asset.get(asset)
        # Prefer 180d; fall back to nearest available
        if prev is None or abs(window - 180) < abs(prev["window_days"] - 180):
            by_asset[asset] = entry
    return by_asset


def compute_hourly_vol(candles_df) -> float:
    """Hourly log-return std from OHLCV close prices."""
    import numpy as np
    closes = candles_df["close"].dropna().to_numpy(dtype=float)
    if len(closes) < 2:
        raise ValueError("Too few candles for vol computation")
    log_rets = np.log(closes[1:] / closes[:-1])
    return float(log_rets.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute collateral sizing from live HL data")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--horizon", type=float, default=24.0,
                        help="Review horizon in hours (default 24)")
    parser.add_argument("--lgd", type=float, default=0.5,
                        help="Loss given default fraction (default 0.5)")
    parser.add_argument("--ou-params", default=str(REPO_ROOT / "data" / "ou_params.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "collateral_sizing.json"))
    args = parser.parse_args()

    ou_params_path = Path(args.ou_params)
    if not ou_params_path.exists():
        print(f"ERROR: {ou_params_path} not found — run calibrate_ou_funding.py first")
        raise SystemExit(1)

    ou_map = load_ou_params(ou_params_path)
    client = HyperliquidClient()
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - VOL_LOOKBACK_DAYS * MS_PER_DAY

    all_results = []
    vol_summary: dict[str, float] = {}

    for asset in args.assets:
        print(f"\n[{asset}]")
        ou = ou_map.get(asset)
        if ou is None:
            print(f"  SKIP: no OU params for {asset}")
            continue

        # Realized vol from hourly candles
        print(f"  fetching candles ({VOL_LOOKBACK_DAYS}d) ...", end=" ", flush=True)
        try:
            candles = client.fetch_candles(asset, "1h", start_ms, now_ms)
            print(f"{len(candles)} candles")
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            continue

        sigma_h = compute_hourly_vol(candles)
        vol_summary[asset] = sigma_h
        print(f"  σ_h = {sigma_h:.6f}  (annualised ≈ {sigma_h * math.sqrt(8760):.1%})")

        # Carry estimate: use OU long-run mean (theta); skip if non-positive
        kappa_h = ou["theta"]
        print(f"  κ̃_h = {kappa_h:.4e}  (from {ou['window_days']}d OU window)")
        if kappa_h <= 0:
            print(f"  NOTE: theta ≤ 0 — carry is negative; economic sizing skipped")

        # Maintenance margin from venue rules
        l_max = HL_MAX_LEVERAGE.get(asset, 20.0)
        theta_F = theta_F_from_max_leverage(l_max)
        print(f"  θ_F = {theta_F:.4f}  (L_max={l_max:.0f}×)")

        results = size_collateral(
            asset=asset,
            sigma_h=sigma_h,
            theta_F=theta_F,
            kappa_h=max(kappa_h, 0.0),
            eps_levels=EPS_LEVELS,
            stress_mults=STRESS_MULTS,
            h_hours=args.horizon,
            lgd=args.lgd,
        )
        all_results.extend(results)

        for res in results:
            print(f"  {res.one_line()}")

        time.sleep(0.3)

    if not all_results:
        print("\nNo results — check network / OU params.")
        return

    print("\n" + "=" * 80)
    print(results_to_table(all_results))
    print("=" * 80)

    # Serialise
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assets": args.assets,
        "horizon_hours": args.horizon,
        "lgd": args.lgd,
        "vol_lookback_days": VOL_LOOKBACK_DAYS,
        "eps_levels": list(EPS_LEVELS),
        "stress_mults": list(STRESS_MULTS),
        "vol_summary": vol_summary,
        "results": [
            {
                "asset": r.asset,
                "method": r.method,
                "eps": r.eps if not math.isnan(r.eps) else None,
                "stress_mult": r.stress_mult,
                "sigma_h_base": r.sigma_h_base,
                "sigma_h_stressed": r.sigma_h_stressed,
                "kappa_h": r.kappa_h,
                "theta_F": r.theta_F,
                "h_hours": r.h_hours,
                "alpha_star": r.alpha_star,
                "pi_liq": r.pi_liq,
                "r_liq": r.r_liq,
                "carry_net_hourly": r.carry_net_hourly,
                "carry_annual_bps": r.carry_annual_bps,
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
