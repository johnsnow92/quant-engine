"""Reconciliation daemon loop tests — spec docs/plans/07 §4.4 / build step 5.

Drives the periodic safety monitor with injected clock/sleep/flatten so the
loop is exercised with no real venue, no real time, and no real orders. Covers:
healthy tick, delta-drift flatten, margin-breach flatten, the dead-man's-switch,
alert routing, and the run-loop (stop predicate + break-on-flatten).
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_engine.execution.recon_daemon import ReconDaemon
from quant_engine.execution.reconcile import LegSnapshot


@dataclass
class _FakeReader:
    """Returns a fixed leg snapshot — stands in for a read-only adapter."""
    venue: str
    position_btc: float
    margin_buffer_pct: float

    def leg_snapshot(self, instrument: str) -> LegSnapshot:
        return LegSnapshot(self.venue, self.position_btc, self.margin_buffer_pct)


class _Clock:
    """Injected clock: now() reads, sleep() advances. No wall-clock time."""
    def __init__(self, start: float = 1000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _daemon(long_btc=0.04, short_btc=-0.04, long_buf=30.0, short_buf=30.0, clock=None):
    flat_calls: list[str] = []
    alert_calls: list[str] = []
    clock = clock or _Clock()
    daemon = ReconDaemon(
        long_reader=_FakeReader("kalshi-perp", long_btc, long_buf),
        short_reader=_FakeReader("coinbase-futures", short_btc, short_buf),
        flatten_fn=flat_calls.append,
        now_fn=clock.now,
        alert_fn=alert_calls.append,
    )
    return daemon, flat_calls, alert_calls, clock


# ---------------------------------------------------------------------------
# Single tick
# ---------------------------------------------------------------------------

def test_healthy_tick_does_not_flatten():
    daemon, flats, _, clock = _daemon()
    tick = daemon.tick(last_recon_epoch=clock.now())
    assert tick.ok
    assert tick.flattened is False
    assert flats == []


def test_delta_drift_flattens():
    daemon, flats, alerts, clock = _daemon(short_btc=-0.05)   # net -0.01 BTC
    tick = daemon.tick(last_recon_epoch=clock.now())
    assert tick.flattened
    assert "delta drift" in tick.reason
    assert len(flats) == 1
    assert len(alerts) == 1


def test_margin_breach_flattens():
    daemon, flats, _, clock = _daemon(short_buf=5.0)   # below the 10% floor
    tick = daemon.tick(last_recon_epoch=clock.now())
    assert tick.flattened
    assert "coinbase-futures margin buffer" in tick.reason
    assert len(flats) == 1


def test_dead_mans_switch_flattens_even_when_healthy():
    # Position is perfectly healthy, but the last recon was 300s ago (> 120s max).
    clock = _Clock(start=1000.0)
    daemon, flats, _, _ = _daemon(clock=clock)
    tick = daemon.tick(last_recon_epoch=700.0)   # now=1000 → 300s stale
    assert tick.flattened
    assert "heartbeat stale" in tick.reason
    assert len(flats) == 1


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def _stop_after(n: int):
    state = {"i": 0}

    def should_stop() -> bool:
        if state["i"] >= n:
            return True
        state["i"] += 1
        return False

    return should_stop


def test_run_loop_monitors_until_stop():
    clock = _Clock()
    daemon, flats, _, _ = _daemon(clock=clock)
    ticks = daemon.run(
        interval_s=30.0,
        sleep_fn=clock.sleep,
        should_stop=_stop_after(3),
    )
    assert len(ticks) == 3
    assert all(t.ok for t in ticks)
    assert flats == []
    # Three 30s sleeps advanced the injected clock.
    assert clock.t == 1000.0 + 3 * 30.0


def test_run_loop_breaks_on_flatten():
    clock = _Clock()
    daemon, flats, _, _ = _daemon(short_btc=-0.06, clock=clock)   # drift → flatten
    ticks = daemon.run(
        interval_s=30.0,
        sleep_fn=clock.sleep,
        should_stop=_stop_after(10),
    )
    assert len(ticks) == 1          # broke after the first flatten
    assert ticks[0].flattened
    assert len(flats) == 1


# ---------------------------------------------------------------------------
# Fail-safety (Codex review): venue read failure + flatten failure
# ---------------------------------------------------------------------------

def test_snapshot_read_failure_flattens_defensively():
    class _BoomReader:
        def leg_snapshot(self, instrument):
            raise ConnectionError("venue API timeout")

    clock = _Clock()
    flats: list[str] = []
    daemon = ReconDaemon(
        long_reader=_BoomReader(),
        short_reader=_FakeReader("coinbase-futures", -0.04, 30.0),
        flatten_fn=flats.append,
        now_fn=clock.now,
    )
    tick = daemon.tick(last_recon_epoch=clock.now())
    assert tick.flattened is True
    assert "snapshot read failed" in tick.reason
    assert len(flats) == 1


def test_failed_flatten_does_not_crash_and_reports_unflattened():
    clock = _Clock()
    alerts: list[str] = []

    def boom(reason: str):
        raise RuntimeError("flatten venue down")

    daemon = ReconDaemon(
        long_reader=_FakeReader("kalshi-perp", 0.04, 30.0),
        short_reader=_FakeReader("coinbase-futures", -0.06, 30.0),   # delta drift → breach
        flatten_fn=boom,
        now_fn=clock.now,
        alert_fn=alerts.append,
    )
    tick = daemon.tick(last_recon_epoch=clock.now())
    assert tick.ok is False
    assert tick.flattened is False                       # flatten did not complete
    assert any("FLATTEN FAILED" in a for a in alerts)


def test_failed_flatten_keeps_retrying_until_flat():
    clock = _Clock()
    attempts = {"n": 0}

    def flaky(reason: str):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("flatten venue down")
        # third attempt succeeds (no raise)

    daemon = ReconDaemon(
        long_reader=_FakeReader("kalshi-perp", 0.04, 30.0),
        short_reader=_FakeReader("coinbase-futures", -0.06, 30.0),
        flatten_fn=flaky,
        now_fn=clock.now,
    )
    ticks = daemon.run(interval_s=30.0, sleep_fn=clock.sleep, should_stop=_stop_after(10))
    assert attempts["n"] == 3            # retried through two failures
    assert ticks[-1].flattened is True   # broke only once the flatten completed
    assert len(ticks) == 3
