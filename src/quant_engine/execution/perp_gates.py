"""Perp pre-trade gates — spec docs/plans/07-perp-executor.md §4.2.

Every proposed two-leg entry must pass ALL gates before the atomic executor places
anything. These are pure, deterministic functions (no LLM, no I/O) over a
PerpTradeProposal + PerpGateConfig; each returns ``(ok, reason)``. ``check_all``
runs them all and returns a GateResult listing every failure.

Gate #1 (allowlist) is the hard legality gate — only ``kalshi-perp`` and
``coinbase-futures`` may receive an order; Coinbase INTX and everything else are
rejected by default-deny.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The only two venues the perp executor may route to (spec §2). Default-deny.
PERP_VENUES = frozenset({"kalshi-perp", "coinbase-futures"})

_HOURS_PER_YEAR = 8_760.0


@dataclass(frozen=True)
class PerpTradeProposal:
    """A proposed delta-neutral two-leg entry. Notionals are USD, qty is BTC."""

    # Long leg (the ~0%-funding leg, e.g. Kalshi perp)
    long_venue: str
    long_qty_btc: float
    long_notional_usd: float
    long_leverage: float
    long_liq_buffer_pct: float          # % above maintenance margin at entry

    # Short leg (the +funding leg, e.g. Coinbase CFM perp)
    short_venue: str
    short_qty_btc: float
    short_notional_usd: float
    short_leverage: float
    short_liq_buffer_pct: float

    # Economics
    long_funding_annual: float          # signed annualized funding on the long leg (we PAY it)
    short_funding_annual: float         # signed annualized funding on the short leg (we RECEIVE it)
    funding_diff_annual: float          # net annualized carry edge (short - long); informational
    hold_hours: float                   # expected hold, for the edge projection
    round_trip_fees_usd: float          # entry + exit fees, both legs


@dataclass(frozen=True)
class PerpGateConfig:
    net_delta_eps_btc: float = 0.001
    kalshi_max_leverage: float = 3.0
    coinbase_max_leverage: float = 3.0
    min_edge_buffer_usd: float = 5.0    # projected funding must clear fees by this much
    min_liq_buffer_pct: float = 20.0
    daily_loss_cap_usd: float = 200.0
    max_leg_notional_usd: float = 2_500.0


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Individual gates (pure)
# --------------------------------------------------------------------------

def gate_allowlist(p: PerpTradeProposal, cfg: PerpGateConfig) -> tuple[bool, str]:
    for venue in (p.long_venue, p.short_venue):
        if venue.strip().lower() not in PERP_VENUES:
            return False, f"venue {venue!r} not in perp allowlist {sorted(PERP_VENUES)}"
    return True, ""


def gate_net_delta(p: PerpTradeProposal, cfg: PerpGateConfig) -> tuple[bool, str]:
    net = p.long_qty_btc - p.short_qty_btc   # short hedges the long
    if abs(net) > cfg.net_delta_eps_btc:
        return False, f"net delta {net:+.6f} BTC exceeds eps {cfg.net_delta_eps_btc}"
    return True, ""


def gate_leverage(p: PerpTradeProposal, cfg: PerpGateConfig) -> tuple[bool, str]:
    if p.long_leverage > cfg.kalshi_max_leverage:
        return False, f"long leverage {p.long_leverage} > cap {cfg.kalshi_max_leverage}"
    if p.short_leverage > cfg.coinbase_max_leverage:
        return False, f"short leverage {p.short_leverage} > cap {cfg.coinbase_max_leverage}"
    return True, ""


def gate_edge_clears_fees(p: PerpTradeProposal, cfg: PerpGateConfig) -> tuple[bool, str]:
    # Per-leg funding (Codex live-path req #4): we RECEIVE on the short leg and PAY
    # on the long leg, each on its OWN notional. The old long-notional proxy
    # (long_notional * funding_diff) is exact only when the leg notionals match;
    # once live marks diverge they don't, so compute the two legs separately.
    projected = (
        p.short_notional_usd * p.short_funding_annual
        - p.long_notional_usd * p.long_funding_annual
    ) * (p.hold_hours / _HOURS_PER_YEAR)
    required = p.round_trip_fees_usd + cfg.min_edge_buffer_usd
    if projected < required:
        return False, (
            f"projected funding ${projected:.2f} < fees ${p.round_trip_fees_usd:.2f} "
            f"+ buffer ${cfg.min_edge_buffer_usd:.2f}"
        )
    return True, ""


def gate_liquidation_buffer(p: PerpTradeProposal, cfg: PerpGateConfig) -> tuple[bool, str]:
    if p.long_liq_buffer_pct < cfg.min_liq_buffer_pct:
        return False, f"long liq buffer {p.long_liq_buffer_pct}% < {cfg.min_liq_buffer_pct}%"
    if p.short_liq_buffer_pct < cfg.min_liq_buffer_pct:
        return False, f"short liq buffer {p.short_liq_buffer_pct}% < {cfg.min_liq_buffer_pct}%"
    return True, ""


def gate_daily_loss(
    p: PerpTradeProposal, cfg: PerpGateConfig, day_pnl_usd: float
) -> tuple[bool, str]:
    if day_pnl_usd <= -cfg.daily_loss_cap_usd:
        return False, f"daily P&L ${day_pnl_usd:.2f} hit kill cap -${cfg.daily_loss_cap_usd:.2f}"
    return True, ""


def gate_max_position(p: PerpTradeProposal, cfg: PerpGateConfig) -> tuple[bool, str]:
    if p.long_notional_usd > cfg.max_leg_notional_usd:
        return False, f"long notional ${p.long_notional_usd} > cap ${cfg.max_leg_notional_usd}"
    if p.short_notional_usd > cfg.max_leg_notional_usd:
        return False, f"short notional ${p.short_notional_usd} > cap ${cfg.max_leg_notional_usd}"
    return True, ""


def check_all(
    p: PerpTradeProposal,
    cfg: PerpGateConfig | None = None,
    day_pnl_usd: float = 0.0,
) -> GateResult:
    """Run every gate. An entry is allowed only if all pass."""
    cfg = cfg or PerpGateConfig()
    checks = [
        gate_allowlist(p, cfg),
        gate_net_delta(p, cfg),
        gate_leverage(p, cfg),
        gate_edge_clears_fees(p, cfg),
        gate_liquidation_buffer(p, cfg),
        gate_daily_loss(p, cfg, day_pnl_usd),
        gate_max_position(p, cfg),
    ]
    failures = [reason for ok, reason in checks if not ok]
    return GateResult(passed=not failures, failures=failures)
