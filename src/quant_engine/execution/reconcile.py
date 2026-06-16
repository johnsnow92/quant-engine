"""Reconciliation logic for the perp carry position — spec docs/plans/07 §4.4.

Continuous safety net. Given a fresh snapshot of both legs (signed positions +
margin buffers), decide whether the position is still safe — delta-neutral within
eps and both legs above the margin floor — and whether to AUTO-FLATTEN. Plus the
dead-man's-switch: if the daemon stops heart-beating, flatten.

Pure + deterministic (no I/O, no LLM). The daemon loop that pulls snapshots from
the venues and calls flatten wraps these decisions; the decisions are tested here
without any live venue. Watch the Coinbase 4pm ET overnight-margin step-up — it
shows up as a margin-buffer drop the reconciler catches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegSnapshot:
    venue: str
    position_btc: float          # signed: + long, - short
    margin_buffer_pct: float     # % above maintenance margin right now


@dataclass(frozen=True)
class PositionSnapshot:
    long: LegSnapshot
    short: LegSnapshot
    funding_accrued_usd: float = 0.0   # net realized funding so far (informational)


@dataclass(frozen=True)
class ReconConfig:
    net_delta_eps_btc: float = 0.001
    min_margin_buffer_pct: float = 10.0    # flatten if either leg drops below this
    heartbeat_max_seconds: float = 120.0


@dataclass
class ReconResult:
    ok: bool
    breaches: list[str] = field(default_factory=list)
    should_flatten: bool = False


def net_delta(snap: PositionSnapshot) -> float:
    """Signed net BTC delta across the two legs (≈0 when delta-neutral)."""
    return snap.long.position_btc + snap.short.position_btc


def reconcile(snap: PositionSnapshot, cfg: ReconConfig | None = None) -> ReconResult:
    """Decide whether the live position is safe; any breach => auto-flatten."""
    cfg = cfg or ReconConfig()
    breaches: list[str] = []

    # Non-finite data must FLATTEN, never pass: a NaN/inf compares False against
    # every threshold, so an unvalidated malformed payload would read as healthy —
    # the exact false-positive the reconciler exists to catch.
    nd = net_delta(snap)
    if not math.isfinite(nd):
        breaches.append(f"non-finite net delta ({nd}) — malformed position data, flattening")
    elif abs(nd) > cfg.net_delta_eps_btc:
        breaches.append(f"delta drift {nd:+.6f} BTC exceeds eps {cfg.net_delta_eps_btc}")

    for leg in (snap.long, snap.short):
        if not math.isfinite(leg.margin_buffer_pct):
            breaches.append(
                f"{leg.venue} non-finite margin buffer ({leg.margin_buffer_pct}) "
                f"— malformed data, flattening"
            )
        elif leg.margin_buffer_pct < cfg.min_margin_buffer_pct:
            breaches.append(
                f"{leg.venue} margin buffer {leg.margin_buffer_pct}% "
                f"below floor {cfg.min_margin_buffer_pct}%"
            )

    return ReconResult(ok=not breaches, breaches=breaches, should_flatten=bool(breaches))


def heartbeat_stale(
    last_recon_epoch: float, now_epoch: float, cfg: ReconConfig | None = None
) -> bool:
    """Dead-man's-switch: True if no recon write within heartbeat_max → flatten.

    Fail-closed on non-finite timing: a NaN epoch/config would make the
    comparison False and silently suppress the flatten, so treat it as stale.
    """
    cfg = cfg or ReconConfig()
    if (
        not math.isfinite(last_recon_epoch)
        or not math.isfinite(now_epoch)
        or not math.isfinite(cfg.heartbeat_max_seconds)
        or cfg.heartbeat_max_seconds <= 0.0
    ):
        return True
    return (now_epoch - last_recon_epoch) > cfg.heartbeat_max_seconds
