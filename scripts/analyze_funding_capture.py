"""Measure empirical funding capture for the short-perp carry leg.

The backtest assumed a flat 90% funding-capture haircut. That number had no
mechanistic basis: Hyperliquid pays funding hourly and mechanically, so a short
receives exactly the funding rate × notional with no spread. The genuine drag
is the hours when funding goes NEGATIVE (you are short and must pay longs) —
and that is already present in the signed funding series the backtest sums.

This script measures, from the real hourly funding history:
  - the share of hours funding is positive / negative / zero,
  - gross positive funding income vs negative funding paid,
  - the empirical capture ratio = net / positive-only (the fraction of the
    best-case "short only when paid" income that survives the negative hours),
  - the residual execution friction (≈0 on HL — funding is paid mechanically).

The capture ratio is α-independent (the (1-α) notional scaling cancels), so it
is a clean property of the funding series itself.

Usage:
    uv run python scripts/analyze_funding_capture.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.hyperliquid import HyperliquidClient

MS_PER_DAY = 86_400_000
HISTORY_DAYS = 400
HOURS_PER_YEAR = 8760.0
ASSETS = ["ETH", "BTC", "SOL"]


def fetch_with_retry(fn, *args, attempts: int = 5, **kwargs):
    last = None
    for k in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — network flakiness, retry
            last = exc
            time.sleep(1.0 + k)
    raise last


def main() -> None:
    client = HyperliquidClient()
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - HISTORY_DAYS * MS_PER_DAY

    print(f"{'='*72}")
    print("Empirical funding capture — Hyperliquid hourly funding")
    print(f"{'='*72}")

    for asset in ASSETS:
        try:
            fdf = fetch_with_retry(client.fetch_funding_history, asset, start_ms, now_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{asset}] fetch failed: {exc}")
            continue

        f = fdf["funding_rate"].to_numpy(dtype=float)
        n = len(f)
        if n == 0:
            print(f"\n[{asset}] no funding records")
            continue

        pos = f[f > 0]
        neg = f[f < 0]
        zero_n = int(np.sum(f == 0))

        mean_all = float(f.mean())                      # net per-hour funding (signed)
        gross_pos = float(pos.sum())                    # total received (positive hours)
        gross_neg = float(neg.sum())                    # total paid (negative hours, ≤0)
        net = float(f.sum())                            # received + paid

        # Annualised bps (per unit notional; the (1-α) scaling is a level factor)
        net_bps = mean_all * HOURS_PER_YEAR * 1e4
        pos_only_ann_bps = (gross_pos / n) * HOURS_PER_YEAR * 1e4
        neg_ann_bps = (gross_neg / n) * HOURS_PER_YEAR * 1e4

        capture = net / gross_pos if gross_pos > 0 else float("nan")

        print(f"\n[{asset}]  {n} hourly funding records ({n/24:.0f} days)")
        print(f"  funding sign:   +{len(pos)/n:.1%} positive   "
              f"{len(neg)/n:.1%} negative   {zero_n/n:.1%} zero")
        print(f"  mean funding (signed):      {mean_all:.3e}/hr  "
              f"→ net {net_bps:+.0f} bps/yr")
        print(f"  positive-only income:       {pos_only_ann_bps:+.0f} bps/yr "
              f"(short only when paid)")
        print(f"  negative-funding paid:      {neg_ann_bps:+.0f} bps/yr "
              f"(drag from short paying longs)")
        print(f"  EMPIRICAL CAPTURE = net / positive-only = {capture:.3f} "
              f"({capture*100:.1f}%)")
        print(f"  → on HL there is no spread on the funding payment itself, so this")
        print(f"     ratio IS the realistic capture; the backtest's signed gross_carry")
        print(f"     already embeds the {1-capture:.1%} negative-funding drag.")
        time.sleep(0.3)

    print(f"\n{'='*72}")
    print("Interpretation: set funding_capture ≈ the measured ratio ONLY if the")
    print("backtest used positive-only funding. Since the backtest sums SIGNED")
    print("funding, negatives are already counted — so the execution-only capture")
    print("knob should be ~1.0, not 0.90. Re-run the backtest with the measured value.")


if __name__ == "__main__":
    main()
