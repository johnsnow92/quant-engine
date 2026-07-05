"""Tests for the funding-regime watcher's alert-gating decision.

The alert gate was previously inlined in main() and untested, and it had a bug:
--always-alert only fired when the regime was already ON, so the workflow's
"Send Telegram regardless of regime state" dispatch sent nothing while OFF
(the common case). These pin the corrected truth table plus the OFF-state
message header so a forced alert never falsely claims the regime is ON.
"""
from __future__ import annotations

from datetime import datetime, timezone

from quant_engine.watchers.funding_regime import RegimeState

import importlib.util
from pathlib import Path

# The watcher lives in scripts/ (not an importable package), so load it by path.
_SPEC = importlib.util.spec_from_file_location(
    "watch_funding_regime",
    Path(__file__).resolve().parents[1] / "scripts" / "watch_funding_regime.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
should_send_alert = _MOD.should_send_alert
build_alert = _MOD.build_alert


def _state(is_on: bool) -> RegimeState:
    return RegimeState(
        is_on=is_on,
        btc_cc_ann=0.0,
        btc_carry_ann=0.0,
        btc_hl_ann=0.0,
        btc_spread_ann=0.0,
        eth_cc_ann=0.0,
        triggered_by=[],
        checked_at=datetime(2026, 6, 17, 13, 0, tzinfo=timezone.utc),
    )


def test_always_alert_fires_even_when_regime_off():
    # The bug this PR fixes: a forced/test alert must send regardless of state.
    assert should_send_alert(
        False, always_alert=True, alert_on_change_only=False, prev_on=None
    ) is True


def test_always_alert_fires_when_on():
    assert should_send_alert(
        True, always_alert=True, alert_on_change_only=True, prev_on=True
    ) is True


def test_off_and_not_forced_is_silent():
    assert should_send_alert(
        False, always_alert=False, alert_on_change_only=False, prev_on=None
    ) is False


def test_default_alerts_whenever_on():
    assert should_send_alert(
        True, always_alert=False, alert_on_change_only=False, prev_on=None
    ) is True


def test_change_only_fires_on_off_to_on_transition():
    assert should_send_alert(
        True, always_alert=False, alert_on_change_only=True, prev_on=False
    ) is True


def test_change_only_suppresses_when_already_on():
    assert should_send_alert(
        True, always_alert=False, alert_on_change_only=True, prev_on=True
    ) is False


def test_change_only_fires_when_prior_unknown():
    assert should_send_alert(
        True, always_alert=False, alert_on_change_only=True, prev_on=None
    ) is True


def test_build_alert_off_header_is_honest():
    msg = build_alert(_state(False), prev_on=None)
    assert "OFF" in msg
    assert "regime ON" not in msg  # must not falsely claim ON while OFF
    assert "Funding regime: OFF" in msg  # summary block rendered


def test_build_alert_on_header():
    msg = build_alert(_state(True), prev_on=False)
    assert "TURNED ON" in msg


def test_send_telegram_failure_surfaces_reason_without_leaking_token(caplog):
    from unittest.mock import patch

    # Stands in for the bot token; the raised HTTPError embeds it in the URL
    # exactly as requests does. The log must never echo it back.
    leak_canary = "do-not-log-this-value"
    raised_url = (
        "400 Client Error: Bad Request for url: "
        f"https://api.telegram.org/bot{leak_canary}/sendMessage"
    )

    class _FakeResp:
        status_code = 400
        reason = "Bad Request"
        text = '{"ok":false,"description":"Bad Request: chat not found"}'

        def json(self):
            return {"ok": False, "description": "Bad Request: chat not found"}

        def raise_for_status(self):
            raise _MOD.requests.HTTPError(raised_url, response=self)

    with patch.object(_MOD.requests, "post", return_value=_FakeResp()):
        with caplog.at_level("WARNING"):
            ok = _MOD.send_telegram(leak_canary, "999", "hi")

    assert ok is False
    assert "chat not found" in caplog.text        # useful reason surfaced
    assert leak_canary not in caplog.text         # token-equivalent never logged
