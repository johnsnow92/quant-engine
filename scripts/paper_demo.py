"""Demonstrate the paper broker and the pre-trade guard layer.

Usage:
    uv run python scripts/paper_demo.py
"""
from __future__ import annotations

from quant_engine.execution.guards import GuardRejection, Order, PreTradeGuard
from quant_engine.execution.paper_broker import PaperBroker


def main() -> None:
    guard = PreTradeGuard(
        allowed_instruments={"BTCUSD-PERP"},
        max_notional_usd=50_000,
        max_position_qty=2.0,
    )
    broker = PaperBroker(guard=guard, cash=100_000)

    fill = broker.submit_order(Order("BTCUSD-PERP", "buy", 0.5, 60_000))
    print(f"Filled: {fill}")
    print(f"Position: {broker.position('BTCUSD-PERP')}  Cash: {broker.cash:,.2f}")
    print(f"Equity @60k mark: {broker.equity({'BTCUSD-PERP': 60_000}):,.2f}")

    print("\nGuard checks (each should reject):")
    for label, order in [
        ("oversized notional", Order("BTCUSD-PERP", "buy", 0.5, 200_000)),
        ("disallowed instrument", Order("DOGEUSD-PERP", "buy", 1.0, 0.10)),
    ]:
        try:
            broker.submit_order(order)
            print(f"  [BUG] {label}: order was NOT rejected")
        except GuardRejection as exc:
            print(f"  rejected {label}: {exc}")

    print(f"\nFinal position unchanged by rejected orders: {broker.position('BTCUSD-PERP')}")


if __name__ == "__main__":
    main()
