"""Dry-run the live carry rebalance — paper-first, NO real orders, NO credentials.

Pulls live data (Coinbase spot prices for vol + the spot leg, Hyperliquid funding
for context), computes the target delta-neutral portfolio with the locked sizing
(ε=1%, 2× stress), plans the exact orders to reach it from the current position,
and pushes them through a guard-enforced PaperBroker to prove the path reaches
the target with zero real capital.

The perp mark is proxied by the Coinbase spot price here (basis is negligible for
planning); the live broker stage will use the real Hyperliquid mark.

Usage:
    uv run python scripts/plan_live_rebalance.py
    uv run python scripts/plan_live_rebalance.py --assets ETH --equity 25000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.coinbase import CoinbaseClient
from quant_engine.data.hyperliquid import HyperliquidClient
from quant_engine.execution.target import ExecutionConfig, compute_target
from quant_engine.execution.planner import PositionState, plan_rebalance, dry_run_plan
from quant_engine.execution.guards import PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker

MS_PER_DAY = 86_400_000
VOL_DAYS = 40  # > 30d lookback + buffer
HL_MAX_LEVERAGE = {"ETH": 50.0, "BTC": 50.0}


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
    parser = argparse.ArgumentParser(description="Dry-run the live carry rebalance (paper only)")
    parser.add_argument("--assets", nargs="+", default=["ETH", "BTC"])
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="Per-asset account equity in USD (sized independently)")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--stress", type=float, default=2.0)
    parser.add_argument("--max-notional-usd", type=float, default=1_000_000.0,
                        help="Pre-trade guard: max per-order notional")
    parser.add_argument("--max-position-qty", type=float, default=1_000.0,
                        help="Pre-trade guard: max per-instrument position size")
    args = parser.parse_args()

    cb = CoinbaseClient()
    hl = HyperliquidClient()
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - VOL_DAYS * MS_PER_DAY

    try:
        funding_now = {r["coin"]: r["funding_rate"]
                       for _, r in fetch_with_retry(hl.current_funding).iterrows()}
    except Exception:  # noqa: BLE001 — funding is context only, never blocks the plan
        funding_now = {}

    print(f"{'='*72}")
    print(f"LIVE REBALANCE DRY-RUN  (paper only — no orders sent)")
    print(f"  config: ε={args.eps:.0%}  stress={args.stress:.1f}×  equity=${args.equity:,.0f}/asset")
    print(f"{'='*72}")

    for asset in args.assets:
        cfg = ExecutionConfig(
            eps=args.eps,
            stress_mult=args.stress,
            theta_F=1.0 / (2.0 * HL_MAX_LEVERAGE.get(asset, 20.0)),
        )
        prices = fetch_with_retry(cb.fetch_candles, f"{asset}-USD", start_ms, now_ms)
        closes = prices["close"].to_numpy(dtype=float)
        spot_price = float(closes[-1])
        perp_price = spot_price  # proxy; live broker stage will use the HL mark
        fr = funding_now.get(asset)

        target = compute_target(asset, args.equity, closes, spot_price, perp_price, cfg)

        # Plan from a flat starting position (the live broker stage reads real positions)
        current = PositionState()
        plan = plan_rebalance(target, current)

        guard = PreTradeGuard(
            allowed_instruments={f"{asset}-SPOT", f"{asset}-PERP"},
            max_notional_usd=args.max_notional_usd,
            max_position_qty=args.max_position_qty,
        )
        broker = PaperBroker(guard=guard, cash=args.equity, positions={})
        fills = dry_run_plan(plan, broker)

        fr_str = f"{fr*8760*100:+.1f}%/yr" if fr is not None else "n/a"
        print(f"\n[{asset}]  spot=${spot_price:,.2f}  σ_h={target.sigma_h:.4f} "
              f"(ann {target.sigma_h*8760**0.5:.0%})  funding(now)={fr_str}")
        print(f"  TARGET  α*={target.alpha:.3f}  "
              f"spot=${target.spot_notional:,.0f} long  "
              f"perp=${target.perp_notional:,.0f} short  margin=${target.perp_margin:,.0f}")
        print(f"          barrier r_liq=+{target.barrier_move_pct:.2%} move  "
              f"Π_liq={target.pi_liq:.3%}")
        print(f"  ORDERS ({len(plan.orders)}):")
        for o in plan.orders:
            print(f"    {o.side.upper():>4} {o.qty:.5f} {o.instrument:<9} @ ${o.price:,.2f}"
                  f"  (${o.notional:,.0f})")
        if plan.skipped:
            print(f"    skipped (below min notional): {', '.join(plan.skipped)}")
        print(f"  DRY-RUN FILLS → positions: "
              f"{asset}-SPOT={broker.position(f'{asset}-SPOT'):+.5f}  "
              f"{asset}-PERP={broker.position(f'{asset}-PERP'):+.5f}  "
              f"(fees ${sum(f.fee for f in fills):,.2f})")
        net_delta = broker.position(f"{asset}-SPOT") + broker.position(f"{asset}-PERP")
        print(f"  net delta = {net_delta:+.6f} asset units "
              f"({'DELTA-NEUTRAL ✓' if abs(net_delta) < 1e-6 else 'NOT NEUTRAL ✗'})")
        time.sleep(0.2)

    print(f"\n{'='*72}")
    print("No orders were sent. To go live, the next stage wires an authenticated")
    print("Hyperliquid broker behind this same guard + planner, gated by --live.")


if __name__ == "__main__":
    main()
