"""Perp carry strategy core — spec docs/plans/07 §3 / build step 3.

Pure decision layer between the read-only adapters and the pre-trade gates.
Given a snapshot of both venues (funding rates, marks, margin buffers) it decides
whether the dead-band carry is worth entering and, if so, emits a delta-neutral
`PerpTradeProposal` for the gates to validate and the atomic two-leg executor to
place. No I/O, no LLM — deterministic functions only.

Carry sign convention (matches `funding_check`): we are LONG the long leg and
SHORT the short leg, so when a funding rate is positive the long leg PAYS and the
short leg RECEIVES. The captured annualized edge is therefore
``short_funding - long_funding`` — large when the short (Coinbase CFM) pays ~+6%
and the long (Kalshi) is pinned at ~0 in the dead band.

Sizing is delta-neutral in BTC (equal qty on both legs) and defensive: quantity
is sized off the LARGER mark so neither leg's USD notional can exceed the target,
which keeps the max-position gate satisfied even when the two venue marks differ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .guards import Order
from .perp_gates import PerpTradeProposal


@dataclass(frozen=True)
class PerpMarketInputs:
    """A point-in-time read of both venues, assembled from the adapters."""
    instrument: str                 # internal name, e.g. "BTCUSD-PERP"
    long_venue: str                 # the ~0%-funding leg (kalshi-perp)
    short_venue: str                # the +funding leg (coinbase-futures)
    long_funding_annual: float      # signed, annualized
    short_funding_annual: float
    long_mark_usd: float
    short_mark_usd: float
    long_liq_buffer_pct: float
    short_liq_buffer_pct: float


@dataclass(frozen=True)
class PerpStrategyConfig:
    target_leg_notional_usd: float = 2_500.0
    long_leverage: float = 2.5
    short_leverage: float = 2.5
    hold_hours: float = 2_190.0          # ~one quarter, the edge's decay horizon
    round_trip_fees_usd: float = 20.0    # entry + exit, both legs (estimate)
    min_funding_diff_annual: float = 0.03  # pre-registered kill: diff < 3%/yr → no trade


def captured_carry_annual(inputs: PerpMarketInputs) -> float:
    """Net annualized carry we keep: short receives, long pays (signed)."""
    return inputs.short_funding_annual - inputs.long_funding_annual


def _require_finite(name: str, value: float, positive: bool = False) -> None:
    """Reject NaN/inf at the decision boundary.

    A non-finite mark or rate compares False against every threshold, so it would
    poison qty/notional and slip through the gates as a falsely-valid trade. Marks
    must additionally be strictly positive (Order requires price > 0).
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} is non-finite ({value}) — malformed market data")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def build_proposal(
    inputs: PerpMarketInputs, cfg: PerpStrategyConfig | None = None
) -> PerpTradeProposal | None:
    """Emit a delta-neutral proposal, or None when the edge is too thin to trade.

    Returns None (a legitimate no-trade) when the captured carry is below the
    pre-registered threshold. Raises ValueError on malformed market data — a
    non-positive mark can't be sized and must not silently become a no-trade.
    """
    cfg = cfg or PerpStrategyConfig()

    _require_finite("long_mark_usd", inputs.long_mark_usd, positive=True)
    _require_finite("short_mark_usd", inputs.short_mark_usd, positive=True)
    _require_finite("long_funding_annual", inputs.long_funding_annual)
    _require_finite("short_funding_annual", inputs.short_funding_annual)
    _require_finite("long_liq_buffer_pct", inputs.long_liq_buffer_pct)
    _require_finite("short_liq_buffer_pct", inputs.short_liq_buffer_pct)

    edge = captured_carry_annual(inputs)
    if edge < cfg.min_funding_diff_annual:
        return None

    # Delta-neutral: identical BTC quantity on both legs. Size off the larger
    # mark so each leg's notional stays at or below the target (≤ cap).
    reference_mark = max(inputs.long_mark_usd, inputs.short_mark_usd)
    qty_btc = cfg.target_leg_notional_usd / reference_mark

    return PerpTradeProposal(
        long_venue=inputs.long_venue,
        long_qty_btc=qty_btc,
        long_notional_usd=qty_btc * inputs.long_mark_usd,
        long_leverage=cfg.long_leverage,
        long_liq_buffer_pct=inputs.long_liq_buffer_pct,
        short_venue=inputs.short_venue,
        short_qty_btc=qty_btc,
        short_notional_usd=qty_btc * inputs.short_mark_usd,
        short_leverage=cfg.short_leverage,
        short_liq_buffer_pct=inputs.short_liq_buffer_pct,
        funding_diff_annual=edge,
        hold_hours=cfg.hold_hours,
        round_trip_fees_usd=cfg.round_trip_fees_usd,
    )


def build_orders(inputs: PerpMarketInputs, proposal: PerpTradeProposal) -> tuple[Order, Order]:
    """The (long, short) orders for the atomic two-leg executor.

    Long the 0%-funding leg, short the +funding leg, both at their venue mark.
    Returned in long-then-short order — the executor places the long first and
    unwinds it if the short can't be hedged.
    """
    long_order = Order(inputs.instrument, "buy", proposal.long_qty_btc, inputs.long_mark_usd)
    short_order = Order(inputs.instrument, "sell", proposal.short_qty_btc, inputs.short_mark_usd)
    return long_order, short_order
