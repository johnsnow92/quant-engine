"""Reconciliation daemon loop — spec docs/plans/07 §4.4 / build step 5 (driver).

`reconcile.py` is the pure safety decision; this is the periodic driver that
wraps it for autonomous operation. Each tick:

  1. Enforce the dead-man's-switch first — if too long has elapsed since the last
     successful reconciliation, flatten defensively (a stalled monitor must not
     leave a live position unwatched).
  2. Pull a fresh position snapshot from both read-only adapters.
  3. Reconcile; on any breach (delta drift, margin-floor) call the injected
     flatten action + alert.

The clock, sleep, and flatten/alert actions are all injected, so the loop is
fully unit-testable with no real venue, no real time, and no real order
placement. In shadow the flatten action is a logging no-op; in live it routes a
both-legs flatten through the executor. The daemon never decides economics —
only safety.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from .reconcile import (
    PositionSnapshot,
    ReconConfig,
    ReconResult,
    heartbeat_stale,
    reconcile,
)

log = logging.getLogger(__name__)


@dataclass
class ReconTick:
    """The outcome of one daemon cycle."""
    ok: bool
    flattened: bool
    reason: str
    recon_result: ReconResult | None
    epoch: float            # this tick's recon time — feed forward as last_recon_epoch


@dataclass
class ReconDaemon:
    """Periodic safety monitor over an open two-leg perp position.

    ``long_reader`` / ``short_reader`` are the read-only adapters; only
    ``leg_snapshot`` is called. ``flatten_fn`` performs the both-legs flatten
    (a no-op logger in shadow); ``now_fn`` / ``sleep_fn`` are injected so the
    loop is testable without wall-clock time.
    """
    long_reader: object
    short_reader: object
    flatten_fn: Callable[[str], None]
    now_fn: Callable[[], float]
    instrument: str = "BTCUSD-PERP"
    cfg: ReconConfig = field(default_factory=ReconConfig)
    alert_fn: Callable[[str], None] | None = None

    def tick(self, last_recon_epoch: float) -> ReconTick:
        """One reconciliation cycle against the live (or faked) venue snapshots."""
        now = self.now_fn()

        # Dead-man's-switch first: a stale gap means flatten regardless of reads.
        if heartbeat_stale(last_recon_epoch, now, self.cfg):
            reason = f"heartbeat stale: {now - last_recon_epoch:.0f}s since last recon"
            self._flatten(reason)
            return ReconTick(ok=False, flattened=True, reason=reason, recon_result=None, epoch=now)

        snap = PositionSnapshot(
            long=self.long_reader.leg_snapshot(self.instrument),
            short=self.short_reader.leg_snapshot(self.instrument),
        )
        result = reconcile(snap, self.cfg)
        if result.should_flatten:
            reason = "; ".join(result.breaches)
            self._flatten(reason)
            return ReconTick(ok=False, flattened=True, reason=reason, recon_result=result, epoch=now)

        return ReconTick(ok=True, flattened=False, reason="", recon_result=result, epoch=now)

    def run(
        self,
        *,
        interval_s: float,
        sleep_fn: Callable[[float], None],
        should_stop: Callable[[], bool],
        last_recon_epoch: float | None = None,
    ) -> list[ReconTick]:
        """Drive ticks until ``should_stop()`` or a flatten. Returns every tick.

        Breaks immediately after a flatten — the position is flat, so there is
        nothing to monitor until a re-entry restarts the daemon.
        """
        epoch = last_recon_epoch if last_recon_epoch is not None else self.now_fn()
        ticks: list[ReconTick] = []
        while not should_stop():
            tick = self.tick(epoch)
            ticks.append(tick)
            epoch = tick.epoch
            if tick.flattened:
                break
            sleep_fn(interval_s)
        return ticks

    # ------------------------------------------------------------------

    def _flatten(self, reason: str) -> None:
        log.critical("[recon] FLATTEN — %s", reason)
        if self.alert_fn is not None:
            self.alert_fn(f"[recon] auto-flatten: {reason}")
        self.flatten_fn(reason)
