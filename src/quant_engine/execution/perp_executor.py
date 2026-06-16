"""Gated perp executor — the enforcement boundary for the live order path (spec 07 §12).

Wraps the atomic two-leg executor so NO order can be placed unless its pre-trade
gates passed AND both legs target perp-allowlisted venues. This makes Codex
live-path reqs #2 (structural gate enforcement) and #3 (perp-venue fail-closed)
STRUCTURAL — wired into the only execution path — rather than a convention a
future live caller could skip:

    GatedPerpExecutor.execute(proposal, long_order, short_order)
      1. check_all(proposal) must pass                      (req #2)
      2. authorize_leg(long_venue, gates) / (short_venue)   (req #3 + #2)
      3. TwoLegExecutor.execute(...)  — atomic, reduce-only unwind, verify-flat

Pure orchestration — no I/O, no live placement of its own. The brokers it
delegates to are PaperBrokers in shadow and the live perp brokers once those
exist. A refused order raises ``PerpOrderRefused`` and places nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .guards import Order
from .perp_gates import PerpGateConfig, PerpTradeProposal, check_all
from .perp_order_guard import authorize_leg
from .two_leg import TwoLegExecutor, TwoLegResult


@dataclass
class GatedPerpExecutor:
    """Gate + perp-venue authorization in front of the atomic two-leg executor.

    ``long_broker`` / ``short_broker`` implement ``submit_order(Order) -> Fill``
    (PaperBroker in shadow; the live perp brokers in live). ``mode`` is passed
    through to the two-leg executor for logging.
    """

    long_broker: object
    short_broker: object
    gate_cfg: PerpGateConfig = field(default_factory=PerpGateConfig)
    mode: str = "shadow"

    def execute(
        self,
        proposal: PerpTradeProposal,
        long_order: Order,
        short_order: Order,
        day_pnl_usd: float = 0.0,
    ) -> TwoLegResult:
        """Enforce gates + venue, then place atomically. Raises if refused."""
        # 1. Pre-trade gates must pass — structural, not a caller's responsibility.
        gates = check_all(proposal, self.gate_cfg, day_pnl_usd)

        # 2. Both legs must be perp-allowlisted AND carry passing gates. Either
        #    failing raises PerpOrderRefused before any order is placed.
        authorize_leg(proposal.long_venue, gates)
        authorize_leg(proposal.short_venue, gates)

        # 3. Atomic two-leg placement (reduce-only unwind + verify-flat on a
        #    one-legged failure — never holds a naked leg).
        executor = TwoLegExecutor(self.long_broker, self.short_broker, mode=self.mode)
        return executor.execute(long_order, short_order)
