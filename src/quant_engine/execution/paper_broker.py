"""Paper broker — simulated execution behind the pre-trade guard layer.

Fills at the order price (last/mark), tracks per-instrument positions and cash,
and charges a fee on notional. Every order is checked by the guard first, so a
rejected order leaves positions and cash untouched. No live capital ever flows
through here — this is the simulation boundary that must stay green before any
live adapter is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .guards import Order, PreTradeGuard


@dataclass
class Fill:
    instrument: str
    side: str
    qty: float
    price: float
    fee: float


@dataclass
class PaperBroker:
    guard: PreTradeGuard
    cash: float = 100_000.0
    fee_bps: float = 5.0
    positions: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def submit_order(self, order: Order) -> Fill:
        """Validate against guards, then fill and update state. Raises on reject."""
        current = self.positions.get(order.instrument, 0.0)
        self.guard.check(order, current_position=current)  # raises GuardRejection

        fee = order.notional * (self.fee_bps / 1e4)
        self.positions[order.instrument] = current + order.signed_qty
        self.cash -= order.signed_qty * order.price
        self.cash -= fee

        fill = Fill(order.instrument, order.side, order.qty, order.price, fee)
        self.fills.append(fill)
        return fill

    def position(self, instrument: str) -> float:
        return self.positions.get(instrument, 0.0)

    def equity(self, marks: dict[str, float]) -> float:
        """Cash plus mark-to-market value of open positions."""
        mtm = sum(qty * marks.get(inst, 0.0) for inst, qty in self.positions.items())
        return self.cash + mtm
