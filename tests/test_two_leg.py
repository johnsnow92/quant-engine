"""Atomic two-leg executor — the naked-leg guardrail (spec docs/plans/07 §4.3)."""
from __future__ import annotations

from quant_engine.execution.guards import Order, PreTradeGuard
from quant_engine.execution.paper_broker import Fill, PaperBroker
from quant_engine.execution.two_leg import (
    TwoLegExecutor,
    TwoLegOutcome,
    _reverse,
)


def _paper(instrument: str, **guard_kw) -> PaperBroker:
    base = dict(
        allowed_instruments={instrument},
        max_notional_usd=1e12,
        max_position_qty=1e12,
    )
    base.update(guard_kw)
    return PaperBroker(guard=PreTradeGuard(**base))


def test_reverse_flips_side_only():
    o = Order("BTC-PERP", "buy", 0.01, 63_000.0)
    r = _reverse(o)
    assert r.side == "sell"
    assert r.qty == 0.01
    assert r.price == 63_000.0
    assert _reverse(r).side == "buy"


def test_both_legs_fill_opens_delta_neutral():
    longb = _paper("BTC-K")
    shortb = _paper("BTC-CB")
    ex = TwoLegExecutor(longb, shortb, mode="shadow")

    res = ex.execute(
        Order("BTC-K", "buy", 0.01, 63_000.0),
        Order("BTC-CB", "sell", 0.01, 63_010.0),
    )
    assert res.outcome is TwoLegOutcome.BOTH_FILLED
    assert res.is_safe
    assert res.long_fill.price == 63_000.0
    assert res.short_fill.price == 63_010.0
    # Long +qty, short -qty → delta-neutral across the two venues.
    assert longb.position("BTC-K") == 0.01
    assert shortb.position("BTC-CB") == -0.01


def test_long_fails_nothing_placed():
    longb = _paper("OTHER")     # BTC-K not allowed → long rejected
    shortb = _paper("BTC-CB")
    ex = TwoLegExecutor(longb, shortb)

    res = ex.execute(
        Order("BTC-K", "buy", 0.01, 63_000.0),
        Order("BTC-CB", "sell", 0.01, 63_010.0),
    )
    assert res.outcome is TwoLegOutcome.LONG_FAILED
    assert res.is_safe
    assert shortb.fills == []   # short never attempted — no naked leg


def test_short_fails_long_is_unwound_to_flat():
    longb = _paper("BTC-K")
    shortb = _paper("OTHER")    # BTC-CB not allowed → short rejected after long filled
    ex = TwoLegExecutor(longb, shortb)

    res = ex.execute(
        Order("BTC-K", "buy", 0.01, 63_000.0),
        Order("BTC-CB", "sell", 0.01, 63_010.0),
    )
    assert res.outcome is TwoLegOutcome.UNWOUND
    assert res.is_safe
    assert res.unwind_fill is not None
    assert longb.position("BTC-K") == 0.0   # unwound back to flat — not naked


def test_naked_leg_when_short_and_unwind_both_fail():
    """The worst case: long fills, short rejects, and the unwind also fails (venue
    down). The result is NAKED_LEG (not is_safe) so the recon daemon / caller acts."""

    class _LongFillsThenCannotUnwind:
        def __init__(self) -> None:
            self.calls = 0

        def submit_order(self, order: Order) -> Fill:
            self.calls += 1
            if self.calls == 1:
                return Fill(order.instrument, order.side, order.qty, order.price, 0.0)
            raise RuntimeError("venue down — cannot unwind")

    longb = _LongFillsThenCannotUnwind()
    shortb = _paper("OTHER")    # short rejects
    ex = TwoLegExecutor(longb, shortb)

    res = ex.execute(
        Order("BTC-K", "buy", 0.01, 63_000.0),
        Order("BTC-CB", "sell", 0.01, 63_010.0),
    )
    assert res.outcome is TwoLegOutcome.NAKED_LEG
    assert res.is_safe is False
    assert longb.calls == 2     # tried to unwind
    assert "unwind_failed" in res.error
