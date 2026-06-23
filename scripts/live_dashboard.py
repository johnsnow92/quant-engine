"""Live status dashboard for the spot-perp carry engine — PUBLIC data, no credentials.

A read-only snapshot of the strategy's current state, refreshed each run:

  * realized vol (σ_h) from live Kraken spot candles
  * funding regime across venues — Kraken (the deployment venue) and Hyperliquid
    (offshore reference), trailing 7d/30d annualized, pulled live; Coinbase shown
    as a measured 12-month reference (no public unauthenticated funding endpoint)
  * the locked delta-neutral target (ε=1%, 2× stress) sized to the current vol,
    quantized to whole Bitnomial contracts — α, contracts, barrier, Π_liq
  * live gross carry at the current funding regime, plus the modeled full-cycle
    net from the canonical Kraken backtest artifact

No orders, no credentials, no capital. Purely a situational-awareness tool.

Usage:
    uv run python scripts/live_dashboard.py
    uv run python scripts/live_dashboard.py --assets ETH --equity 25000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from quant_engine.data.kraken import KrakenDataClient, CARRY_SYMBOLS
from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.execution.target import (
    ExecutionConfig,
    compute_target,
    quantize_to_contracts,
    BITNOMIAL_CONTRACT_SIZES,
)

MS_PER_DAY = 86_400_000
KRAKEN_US_THETA_F = 0.025
SPOT_TAKER_BPS = 40.0
PERP_FEE_PER_CONTRACT = 0.15
# Carry breakeven guide: below this annualized funding the net (after ~70bp round-trip
# spot cost + negative-funding drag) is marginal-to-negative.
THIN_FUNDING_THRESHOLD = 0.02
# Measured 12-month Coinbase International funding (no live public endpoint).
COINBASE_REF_12MO = {"BTC": 0.0321, "ETH": 0.0151}


def fetch_with_retry(fn, *args, attempts: int = 3, **kwargs):
    last = None
    for k in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — network flakiness, retry
            last = exc
            time.sleep(0.8 + k)
    raise last


def trailing_annual(df, now_ms: int, days: int) -> float | None:
    """Mean hourly funding over the trailing `days`, annualized (×8760)."""
    if df is None or len(df) == 0:
        return None
    lo = now_ms - days * MS_PER_DAY
    ts = df["ts"].to_numpy()
    fr = df["funding_rate"].to_numpy(dtype=float)[ts >= lo]
    if len(fr) == 0:
        return None
    return float(fr.mean()) * 8760.0


def load_backtest_net(path: Path) -> dict:
    """Map asset -> modeled net ann_return at ε=1%/2× from the canonical artifact."""
    out: dict[str, float] = {}
    try:
        payload = json.loads(path.read_text())
        for r in payload.get("results", []):
            if abs(r.get("eps", 0) - 0.01) < 1e-9 and abs(r.get("stress_mult", 0) - 2.0) < 1e-9:
                out[r["asset"]] = r.get("ann_return")
    except Exception:  # noqa: BLE001 — artifact optional
        pass
    return out


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x*100:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live carry-engine status dashboard (public data)")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--stress", type=float, default=2.0)
    parser.add_argument("--theta-f", type=float, default=KRAKEN_US_THETA_F)
    args = parser.parse_args()

    kr = KrakenDataClient()
    hl = HyperliquidClient()
    now_ms = int(time.time() * 1000)
    bt_net = load_backtest_net(REPO_ROOT / "data" / "backtest_results_kraken.json")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    line = "─" * 72
    print(line)
    print(f"  CARRY ENGINE — LIVE STATUS    {stamp}")
    print(f"  venue: Kraken-US (Bitnomial perp + Kraken spot)   "
          f"config: ε={args.eps:.0%} / {args.stress:.1f}× / θ_F={args.theta_f:.3f}   "
          f"equity ${args.equity:,.0f}/asset")
    print(line)

    for asset in args.assets:
        sym = CARRY_SYMBOLS.get(asset)
        contract_size = BITNOMIAL_CONTRACT_SIZES.get(asset)
        if sym is None or contract_size is None:
            print(f"\n{asset}: unsupported\n")
            continue

        # --- live market data ---
        try:
            spot_df = fetch_with_retry(kr.fetch_spot_ohlc, sym["spot"], 60)
            closes = spot_df["spot_close"].to_numpy(dtype=float)
            spot_price = float(closes[-1])
        except Exception as exc:  # noqa: BLE001
            print(f"\n{asset}: spot fetch failed ({exc})\n")
            continue

        try:
            kr_fund = fetch_with_retry(kr.fetch_funding, sym["perp"])
        except Exception:  # noqa: BLE001
            kr_fund = None
        try:
            hl_fund = fetch_with_retry(hl.fetch_funding_history, asset,
                                       now_ms - 31 * MS_PER_DAY, now_ms)
        except Exception:  # noqa: BLE001
            hl_fund = None

        kr_7d, kr_30d = trailing_annual(kr_fund, now_ms, 7), trailing_annual(kr_fund, now_ms, 30)
        hl_7d, hl_30d = trailing_annual(hl_fund, now_ms, 7), trailing_annual(hl_fund, now_ms, 30)

        # --- live target (sized to current vol, quantized to contracts) ---
        cfg = ExecutionConfig(eps=args.eps, stress_mult=args.stress, theta_F=args.theta_f)
        ideal = compute_target(asset, args.equity, closes, spot_price, spot_price, cfg)
        target = quantize_to_contracts(ideal, contract_size)
        n_contracts = round(abs(target.perp_qty) / contract_size)

        # --- carry estimates ---
        gross_now = ((1.0 - target.alpha) * kr_30d) if kr_30d is not None else None
        spot_cost = target.spot_notional * SPOT_TAKER_BPS / 1e4
        perp_cost = n_contracts * PERP_FEE_PER_CONTRACT
        entry_bps = (spot_cost + perp_cost) / args.equity * 1e4 if args.equity else 0.0
        net_modeled = bt_net.get(asset)

        # --- render ---
        print(f"\n{asset}   spot ${spot_price:,.2f}   "
              f"σ_h {target.sigma_h*100:.2f}%/h (ann {target.sigma_h*8760**0.5:.0%})")
        print(f"  funding (annualized):  "
              f"Kraken 7d {fmt_pct(kr_7d)} / 30d {fmt_pct(kr_30d)}   "
              f"Hyperliquid 7d {fmt_pct(hl_7d)} / 30d {fmt_pct(hl_30d)}   "
              f"Coinbase {fmt_pct(COINBASE_REF_12MO.get(asset))} (12mo ref)")
        if n_contracts == 0:
            print(f"  target:  equity too small for 1 contract "
                  f"({contract_size} {asset} ≈ ${contract_size*spot_price:,.0f})")
        else:
            print(f"  target:  α {target.alpha:.3f}   {n_contracts} contracts "
                  f"({abs(target.perp_qty):g} {asset})   barrier +{target.barrier_move_pct:.1%}"
                  f"   Π_liq {target.pi_liq:.2%}")
            print(f"  costs:   entry ~{entry_bps:.0f}bp "
                  f"(spot ${spot_cost:,.0f} @ {SPOT_TAKER_BPS:.0f}bp + perp ${perp_cost:,.2f})")
        carry_line = (f"  carry:   gross @30d funding {fmt_pct(gross_now)}/yr"
                      f"   |   backtest net (full cycle) {fmt_pct(net_modeled)}/yr")
        print(carry_line)

        # warnings
        if kr_30d is not None and kr_30d < THIN_FUNDING_THRESHOLD:
            state = "NEGATIVE" if kr_30d < 0 else "thin"
            print(f"  ⚠ Kraken 30d funding {state} ({fmt_pct(kr_30d)}/yr) — "
                  f"carry currently {'losing' if kr_30d < 0 else 'marginal after costs'}")

    print(f"\n{line}")
    print("  public data only · no credentials · no orders · no capital at risk")
    print("  funding is live (Kraken/HL); net is the modeled full-cycle backtest figure")
    print(line)


if __name__ == "__main__":
    main()
