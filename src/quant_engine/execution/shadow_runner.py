"""Perp carry shadow-mode runner — spec docs/plans/07 §4.6 / build step 7.

Wires the whole machine end-to-end in SHADOW and places zero live orders:

    read both adapters → build_proposal (strategy core) → pre-trade gates →
    on pass, execute through the atomic two-leg executor with PaperBroker on
    BOTH legs (fills nothing real) → reconcile the resulting paper position.

Shadow is structurally enforced, not just a flag: the live adapters
(`KalshiPerpReader`, `CoinbaseCfmReader`) are read-only and expose no
``submit_order``, so the only objects that can ever place an order here are the
internally-constructed PaperBrokers. This is the ≥3-day decision-vs-logger
verification stage that must run clean before any micro capital (graduation
gate, spec §4.6).

Funding-convention verification (§4.5) is deliberately NOT part of the shadow
cycle — it reconciles REALIZED funding cash, which only exists once micro orders
actually post funding. It activates at the micro stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from .guards import PreTradeGuard
from .paper_broker import PaperBroker
from .perp_gates import GateResult, PerpGateConfig, PerpTradeProposal, check_all
from .perp_strategy import (
    PerpMarketInputs,
    PerpStrategyConfig,
    build_orders,
    build_proposal,
)
from .reconcile import LegSnapshot, PositionSnapshot, ReconConfig, ReconResult, reconcile
from .two_leg import TwoLegExecutor, TwoLegResult

log = logging.getLogger(__name__)


# Decision outcomes for one shadow cycle.
NO_TRADE_THIN_EDGE = "no_trade_thin_edge"
BLOCKED_BY_GATES = "blocked_by_gates"
EXECUTED_SHADOW = "executed_shadow"


@dataclass
class ShadowDecision:
    """The full, logged record of one shadow cycle. Places nothing real."""
    action: str
    inputs: PerpMarketInputs
    proposal: PerpTradeProposal | None = None
    gate_result: GateResult | None = None
    two_leg_result: TwoLegResult | None = None
    recon_result: ReconResult | None = None

    def summary(self) -> str:
        """One-line, Telegram-ready summary of the decision."""
        if self.action == NO_TRADE_THIN_EDGE:
            edge = self.proposal.funding_diff_annual if self.proposal else 0.0
            return f"[shadow] no trade — carry {edge:+.2%}/yr below threshold"
        if self.action == BLOCKED_BY_GATES:
            fails = "; ".join(self.gate_result.failures) if self.gate_result else "?"
            return f"[shadow] BLOCKED by gates: {fails}"
        outcome = self.two_leg_result.outcome.value if self.two_leg_result else "?"
        recon = "ok" if (self.recon_result and self.recon_result.ok) else "BREACH"
        carry = self.proposal.funding_diff_annual if self.proposal else 0.0
        return f"[shadow] executed ({outcome}) carry {carry:+.2%}/yr, recon {recon}"


@dataclass
class PerpShadowRunner:
    """Runs shadow cycles over the two read-only adapters.

    ``long_reader`` / ``short_reader`` only ever have their read methods called
    (funding, mark, margin buffer). Order placement is routed exclusively to
    fresh PaperBrokers from ``broker_factory``.
    """
    long_reader: object       # KalshiPerpReader (read-only)
    short_reader: object      # CoinbaseCfmReader (read-only)
    instrument: str = "BTCUSD-PERP"
    strategy_cfg: PerpStrategyConfig = field(default_factory=PerpStrategyConfig)
    gate_cfg: PerpGateConfig = field(default_factory=PerpGateConfig)
    recon_cfg: ReconConfig = field(default_factory=ReconConfig)
    broker_factory: Callable[[], PaperBroker] | None = None

    def read_inputs(self) -> PerpMarketInputs:
        """Assemble the strategy's market snapshot from both read-only adapters."""
        return PerpMarketInputs(
            instrument=self.instrument,
            long_venue="kalshi-perp",
            short_venue="coinbase-futures",
            long_funding_annual=self.long_reader.funding_rate_annual(self.instrument),
            short_funding_annual=self.short_reader.funding_rate_annual(self.instrument),
            long_mark_usd=self.long_reader.mark_price(self.instrument),
            short_mark_usd=self.short_reader.mark_price(self.instrument),
            long_liq_buffer_pct=self.long_reader.margin_buffer_pct(),
            short_liq_buffer_pct=self.short_reader.margin_buffer_pct(),
        )

    def run_cycle(self, day_pnl_usd: float = 0.0) -> ShadowDecision:
        """One full shadow decision: read → propose → gate → shadow-execute → reconcile."""
        inputs = self.read_inputs()

        proposal = build_proposal(inputs, self.strategy_cfg)
        if proposal is None:
            decision = ShadowDecision(action=NO_TRADE_THIN_EDGE, inputs=inputs)
            log.info(decision.summary())
            return decision

        gate_result = check_all(proposal, self.gate_cfg, day_pnl_usd)
        if not gate_result.passed:
            decision = ShadowDecision(
                action=BLOCKED_BY_GATES, inputs=inputs, proposal=proposal, gate_result=gate_result
            )
            log.warning(decision.summary())
            return decision

        long_order, short_order = build_orders(inputs, proposal)
        long_broker = self._make_broker()
        short_broker = self._make_broker()
        executor = TwoLegExecutor(long_broker=long_broker, short_broker=short_broker, mode="shadow")
        two_leg_result = executor.execute(long_order, short_order)

        recon_result = self._reconcile_paper(inputs, long_broker, short_broker)

        decision = ShadowDecision(
            action=EXECUTED_SHADOW,
            inputs=inputs,
            proposal=proposal,
            gate_result=gate_result,
            two_leg_result=two_leg_result,
            recon_result=recon_result,
        )
        log.info(decision.summary())
        return decision

    # ------------------------------------------------------------------

    def _make_broker(self) -> PaperBroker:
        if self.broker_factory is not None:
            return self.broker_factory()
        guard = PreTradeGuard(
            allowed_instruments={self.instrument},
            max_notional_usd=self.gate_cfg.max_leg_notional_usd * 2.0,
            max_position_qty=10.0,
        )
        return PaperBroker(guard=guard)

    def _reconcile_paper(
        self, inputs: PerpMarketInputs, long_broker: PaperBroker, short_broker: PaperBroker
    ) -> ReconResult:
        snap = PositionSnapshot(
            long=LegSnapshot(
                venue="kalshi-perp",
                position_btc=long_broker.position(self.instrument),
                margin_buffer_pct=inputs.long_liq_buffer_pct,
            ),
            short=LegSnapshot(
                venue="coinbase-futures",
                position_btc=short_broker.position(self.instrument),
                margin_buffer_pct=inputs.short_liq_buffer_pct,
            ),
        )
        return reconcile(snap, self.recon_cfg)
