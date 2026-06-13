"""Reconciliation + dead-man's-switch — spec docs/plans/07 §4.4."""
from __future__ import annotations

import pytest

from quant_engine.execution.reconcile import (
    LegSnapshot,
    PositionSnapshot,
    ReconConfig,
    heartbeat_stale,
    net_delta,
    reconcile,
)

CFG = ReconConfig()


def _snap(long_btc=0.04, short_btc=-0.04, long_buf=30.0, short_buf=30.0) -> PositionSnapshot:
    return PositionSnapshot(
        long=LegSnapshot("kalshi-perp", long_btc, long_buf),
        short=LegSnapshot("coinbase-futures", short_btc, short_buf),
    )


def test_net_delta_zero_when_hedged():
    assert net_delta(_snap()) == pytest.approx(0.0)
    assert net_delta(_snap(short_btc=-0.05)) == pytest.approx(-0.01)


def test_healthy_position_is_ok():
    res = reconcile(_snap(), CFG)
    assert res.ok
    assert res.should_flatten is False
    assert res.breaches == []


def test_delta_drift_triggers_flatten():
    res = reconcile(_snap(short_btc=-0.05), CFG)  # net -0.01 BTC > eps
    assert res.ok is False
    assert res.should_flatten is True
    assert any("delta drift" in b for b in res.breaches)


def test_margin_breach_triggers_flatten():
    # Coinbase 4pm overnight margin step-up drops the short leg below the floor.
    res = reconcile(_snap(short_buf=5.0), CFG)
    assert res.should_flatten is True
    assert any("coinbase-futures margin buffer" in b for b in res.breaches)


def test_multiple_breaches_listed():
    res = reconcile(_snap(short_btc=-0.06, long_buf=2.0), CFG)
    assert res.should_flatten is True
    assert len(res.breaches) >= 2


def test_heartbeat_dead_mans_switch():
    assert heartbeat_stale(last_recon_epoch=0.0, now_epoch=200.0, cfg=CFG) is True   # 200s > 120s
    assert heartbeat_stale(last_recon_epoch=100.0, now_epoch=150.0, cfg=CFG) is False  # 50s < 120s
