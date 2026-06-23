"""Dry-run the live Kraken-US carry rebalance — paper-first, NO real orders, NO credentials.

Stage 2 of the Kraken-US build. Mirrors plan_live_rebalance.py (the Hyperliquid
paper path) but for the venue actually chosen:

  * Spot leg  -> Kraken Pro spot (e.g. ETHUSD), taker ~40bp / maker ~25bp
  * Perp leg  -> Bitnomial US perp (e.g. PETHUI), flat ~$0.15/contract, WHOLE
                 contracts only (BTC 0.01, ETH 0.5, SOL 5 per contract)

It pulls live Kraken spot prices (for realized vol + the spot leg), sizes the
locked delta-neutral target (ε=1%, 2× stress), QUANTIZES the perp leg to whole
Bitnomial contracts and matches the spot leg to it, plans the exact orders, and
pushes them through a guard-enforced PaperBroker — proving the path reaches a
delta-neutral target at executable granularity with zero real capital.

The perp mark is proxied by the Kraken spot price here (basis is negligible for
planning); the live broker stage will use the real Bitnomial mark. Funding shown
is the Kraken PF_ hourly proxy for context — the US perp settles funding daily.

Usage:
    uv run python scripts/plan_kraken_rebalance.py
    uv run python scripts/plan_kraken_rebalance.py --assets ETH --equity 25000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_engine.data.kraken import KrakenDataClient, CARRY_SYMBOLS
from quant_engine.execution.target import (
    ExecutionConfig,
    compute_target,
    quantize_to_contracts,
    BITNOMIAL_CONTRACT_SIZES,
)
from quant_engine.execution.planner import PositionState, plan_rebalance, dry_run_plan
from quant_engine.execution.guards import PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker

# Conservative maintenance-margin fraction for Bitnomial US perps (the live value
# is login-gated; the backtest showed zero liquidations across θ_F 0.01–0.05, so a
# mid value is safe). Refine from the contract spec once the account exists.
KRAKEN_US_THETA_F = 0.025
SPOT_TAKER_BPS = 40.0   # Kraken Pro low-tier taker (maker ~25bp with limit orders)
PERP_FEE_PER_CONTRACT = 0.15  # Bitnomial all-in $/contract/side


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
    parser = argparse.ArgumentParser(
        description="Dry-run the live Kraken-US carry rebalance (paper only)")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="Per-asset account equity in USD (sized independently)")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--stress", type=float, default=2.0)
    parser.add_argument("--theta-f", type=float, default=KRAKEN_US_THETA_F)
    parser.add_argument("--max-notional-usd", type=float, default=1_000_000.0,
                        help="Pre-trade guard: max per-order notional")
    parser.add_argument("--max-position-qty", type=float, default=10_000.0,
                        help="Pre-trade guard: max per-instrument position size")
    args = parser.parse_args()

    kr = KrakenDataClient()

    print(f"{'='*74}")
    print("KRAKEN-US REBALANCE DRY-RUN  (paper only — no orders, no credentials)")
    print(f"  config: ε={args.eps:.0%}  stress={args.stress:.1f}×  θ_F={args.theta_f:.3f}  "
          f"equity=${args.equity:,.0f}/asset")
    print(f"  spot=Kraken Pro (~{SPOT_TAKER_BPS:.0f}bp taker)  "
          f"perp=Bitnomial (${PERP_FEE_PER_CONTRACT:.2f}/contract, whole contracts)")
    print(f"{'='*74}")

    for asset in args.assets:
        sym = CARRY_SYMBOLS.get(asset)
        if sym is None:
            print(f"\n[{asset}]  unsupported (not in CARRY_SYMBOLS)")
            continue
        contract_size = BITNOMIAL_CONTRACT_SIZES.get(asset)
        if contract_size is None:
            print(f"\n[{asset}]  no Bitnomial contract size on record")
            continue

        cfg = ExecutionConfig(eps=args.eps, stress_mult=args.stress, theta_F=args.theta_f)

        spot_df = fetch_with_retry(kr.fetch_spot_ohlc, sym["spot"], 60)
        closes = spot_df["spot_close"].to_numpy(dtype=float)
        spot_price = float(closes[-1])
        perp_price = spot_price  # proxy; live broker uses the Bitnomial mark
        try:
            fr = float(fetch_with_retry(kr.fetch_funding, sym["perp"])["funding_rate"].iloc[-1])
        except Exception:  # noqa: BLE001 — context only, never blocks the plan
            fr = None

        ideal = compute_target(asset, args.equity, closes, spot_price, perp_price, cfg)
        target = quantize_to_contracts(ideal, contract_size)
        n_contracts = round(abs(target.perp_qty) / contract_size)

        fr_str = f"{fr*8760*100:+.1f}%/yr" if fr is not None else "n/a"
        print(f"\n[{asset}]  spot=${spot_price:,.2f}  σ_h={target.sigma_h:.4f} "
              f"(ann {target.sigma_h*8760**0.5:.0%})  funding(PF_ proxy)={fr_str}")

        if n_contracts == 0:
            need = contract_size * spot_price / (1.0 - ideal.alpha)
            print(f"  equity too small for 1 contract "
                  f"({contract_size} {asset} = ${contract_size*spot_price:,.0f}); "
                  f"need ≈ ${need:,.0f}+ equity. SKIPPED.")
            continue

        # Residual from snapping the ideal (1-α) spot notional to whole contracts
        resid = target.spot_notional - ideal.spot_notional
        print(f"  TARGET  α*={target.alpha:.3f}  "
              f"perp={n_contracts} contracts ({abs(target.perp_qty):g} {asset} short)  "
              f"spot=${target.spot_notional:,.0f} long  margin=${target.perp_margin:,.0f}")
        print(f"          ideal spot=${ideal.spot_notional:,.0f} → quantization residual "
              f"${resid:+,.0f} ({resid/args.equity:+.2%} of equity)")
        print(f"          barrier r_liq=+{target.barrier_move_pct:.2%} move  "
              f"Π_liq={target.pi_liq:.3%}")

        current = PositionState()  # flat; live broker stage reads real positions
        plan = plan_rebalance(target, current)

        guard = PreTradeGuard(
            allowed_instruments={f"{asset}-SPOT", f"{asset}-PERP"},
            max_notional_usd=args.max_notional_usd,
            max_position_qty=args.max_position_qty,
        )
        broker = PaperBroker(guard=guard, cash=args.equity, positions={})
        fills = dry_run_plan(plan, broker)

        # Expected real costs at this venue (separate from the paper broker's model)
        spot_cost = target.spot_notional * SPOT_TAKER_BPS / 1e4
        perp_cost = n_contracts * PERP_FEE_PER_CONTRACT
        print(f"  ORDERS ({len(plan.orders)}):")
        for o in plan.orders:
            venue = "Kraken Pro" if o.instrument.endswith("SPOT") else f"Bitnomial {sym['perp']}"
            print(f"    {o.side.upper():>4} {o.qty:g} {o.instrument:<9} @ ${o.price:,.2f}"
                  f"  (${o.notional:,.0f})  [{venue}]")
        if plan.skipped:
            print(f"    skipped (below min notional): {', '.join(plan.skipped)}")
        print(f"  DRY-RUN FILLS → positions: "
              f"{asset}-SPOT={broker.position(f'{asset}-SPOT'):+g}  "
              f"{asset}-PERP={broker.position(f'{asset}-PERP'):+g}")
        print(f"  expected entry cost: spot ${spot_cost:,.2f} ({SPOT_TAKER_BPS:.0f}bp) "
              f"+ perp ${perp_cost:,.2f} ({n_contracts}×${PERP_FEE_PER_CONTRACT:.2f}) "
              f"= ${spot_cost+perp_cost:,.2f} ({(spot_cost+perp_cost)/args.equity*1e4:.1f}bp)")
        net_delta = broker.position(f"{asset}-SPOT") + broker.position(f"{asset}-PERP")
        print(f"  net delta = {net_delta:+.6f} {asset} "
              f"({'DELTA-NEUTRAL ✓' if abs(net_delta) < 1e-9 else 'NOT NEUTRAL ✗'})")
        time.sleep(0.2)

    print(f"\n{'='*74}")
    print("No orders were sent. To go live (Stage 3), the next step wires an")
    print("authenticated Bitnomial perp broker + Kraken spot broker behind this same")
    print("guard + planner, gated by --live. Order TIF must be IOC/Day (no GTC on perps).")


if __name__ == "__main__":
    main()
