"""Order planner — reconcile the live position against the target portfolio.

Given a TargetPortfolio (what we want to hold) and the current PositionState
(what we actually hold), produce the delta-neutral set of orders that closes the
gap, each expressed as a guard-checkable Order. Legs whose adjustment is below a
minimum trade notional are skipped (no dust orders).

The planner never touches the network. dry_run_plan() pushes the resulting
orders through a PaperBroker (which enforces the PreTradeGuard), so the whole
"live equity → exact orders → simulated fills" path is validated with zero real
capital before any live adapter is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .guards import Order
from .paper_broker import Fill, PaperBroker
from .target import TargetPortfolio

SPOT_SUFFIX = "-SPOT"
PERP_SUFFIX = "-PERP"


@dataclass
class PositionState:
    """Current holdings, in asset units. perp_qty is negative when short."""
    spot_qty: float = 0.0
    perp_qty: float = 0.0


@dataclass
class RebalancePlan:
    target: TargetPortfolio
    current: PositionState
    spot_delta_qty: float          # target.spot_qty − current.spot_qty
    perp_delta_qty: float          # target.perp_qty − current.perp_qty
    orders: list[Order]
    skipped: list[str] = field(default_factory=list)  # legs below min trade notional

    @property
    def is_noop(self) -> bool:
        return not self.orders


def _spot_instrument(asset: str) -> str:
    return f"{asset}{SPOT_SUFFIX}"


def _perp_instrument(asset: str) -> str:
    return f"{asset}{PERP_SUFFIX}"


def plan_rebalance(
    target: TargetPortfolio,
    current: PositionState | None = None,
    min_trade_notional: float = 10.0,
) -> RebalancePlan:
    """Build the orders that move ``current`` to ``target``.

    Spot leg trades at target.spot_price, perp leg at target.perp_price. A leg
    whose |Δqty|·price is below ``min_trade_notional`` is skipped to avoid dust.
    """
    current = current or PositionState()
    spot_delta = target.spot_qty - current.spot_qty
    perp_delta = target.perp_qty - current.perp_qty

    orders: list[Order] = []
    skipped: list[str] = []

    if abs(spot_delta) * target.spot_price >= min_trade_notional:
        orders.append(Order(
            instrument=_spot_instrument(target.asset),
            side="buy" if spot_delta > 0 else "sell",
            qty=abs(spot_delta),
            price=target.spot_price,
        ))
    else:
        skipped.append("spot")

    if abs(perp_delta) * target.perp_price >= min_trade_notional:
        orders.append(Order(
            instrument=_perp_instrument(target.asset),
            side="buy" if perp_delta > 0 else "sell",  # +Δ reduces a short, −Δ grows it
            qty=abs(perp_delta),
            price=target.perp_price,
        ))
    else:
        skipped.append("perp")

    return RebalancePlan(
        target=target,
        current=current,
        spot_delta_qty=spot_delta,
        perp_delta_qty=perp_delta,
        orders=orders,
        skipped=skipped,
    )


def dry_run_plan(plan: RebalancePlan, broker: PaperBroker) -> list[Fill]:
    """Submit the plan's orders through a PaperBroker (guard-enforced, simulated).

    The broker should be seeded with the current positions so the resulting
    positions can be checked against the target. Raises GuardRejection if any
    order violates a pre-trade limit.
    """
    return [broker.submit_order(order) for order in plan.orders]
