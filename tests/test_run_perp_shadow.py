"""Unit tests for scripts/run_perp_shadow.py — graceful cron behavior, no live calls.

The shadow machinery itself is exhaustively covered elsewhere; these tests pin the
script's glue: it must no-op gracefully when read credentials are absent (the
expected default until [OP] provisions them), wire the readers into the runner
when they are present, and never let a read/endpoint hiccup fail the scheduled job.
"""
from __future__ import annotations

import importlib.util
import pathlib
from unittest.mock import MagicMock, patch

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_perp_shadow.py"
_spec = importlib.util.spec_from_file_location("run_perp_shadow", _SCRIPT)
run_perp_shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_perp_shadow)

_ALL_CREDS = {
    "KALSHI_PERP_KEY_ID": "k",
    "KALSHI_PERP_PRIVATE_KEY": "kpem",
    "COINBASE_CFM_API_KEY_NAME": "c",
    "COINBASE_CFM_PRIVATE_KEY": "cpem",
}
_PATCH_KALSHI = "quant_engine.execution.kalshi_perp_broker.KalshiPerpReader"
_PATCH_CFM = "quant_engine.execution.coinbase_cfm_broker.CoinbaseCfmReader"
_PATCH_RUNNER = "quant_engine.execution.shadow_runner.PerpShadowRunner"


@pytest.fixture(autouse=True)
def _argv(monkeypatch):
    # argparse reads sys.argv; pin it so pytest's own argv doesn't leak in.
    monkeypatch.setattr("sys.argv", ["run_perp_shadow.py"])


def _clear_creds(monkeypatch):
    for name in (*run_perp_shadow._KALSHI_ENV, *run_perp_shadow._CFM_ENV,
                 "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)


def _set_creds(monkeypatch):
    for key, val in _ALL_CREDS.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


def test_missing_credentials_lists_all_unset(monkeypatch):
    _clear_creds(monkeypatch)
    assert set(run_perp_shadow.missing_credentials()) == set(
        (*run_perp_shadow._KALSHI_ENV, *run_perp_shadow._CFM_ENV)
    )


def test_no_creds_exits_zero_and_blocks(monkeypatch, caplog):
    _clear_creds(monkeypatch)
    with patch(_PATCH_KALSHI) as kalshi, caplog.at_level("INFO"):
        rc = run_perp_shadow.main()
    assert rc == 0
    assert "BLOCKED" in caplog.text
    kalshi.assert_not_called()          # never reached the live readers


def test_with_creds_runs_cycle_and_logs_summary(monkeypatch, caplog):
    _set_creds(monkeypatch)
    from quant_engine.execution.shadow_runner import EXECUTED_SHADOW

    decision = MagicMock()
    decision.summary.return_value = "[shadow] executed (both_filled) carry +6.00%/yr, recon ok"
    decision.action = EXECUTED_SHADOW
    fake_runner = MagicMock()
    fake_runner.run_cycle.return_value = decision

    with patch(_PATCH_KALSHI), patch(_PATCH_CFM), \
            patch(_PATCH_RUNNER, return_value=fake_runner), caplog.at_level("INFO"):
        rc = run_perp_shadow.main()

    assert rc == 0
    fake_runner.run_cycle.assert_called_once()
    assert "executed" in caplog.text


def test_read_failure_exits_zero(monkeypatch, caplog):
    _set_creds(monkeypatch)
    fake_runner = MagicMock()
    fake_runner.run_cycle.side_effect = RuntimeError("404 from confirm-at-micro endpoint")

    with patch(_PATCH_KALSHI), patch(_PATCH_CFM), \
            patch(_PATCH_RUNNER, return_value=fake_runner), caplog.at_level("WARNING"):
        rc = run_perp_shadow.main()

    assert rc == 0
    assert "read/cycle failed" in caplog.text


def test_send_telegram_logs_non_200(caplog):
    # A bad token / rate limit (non-200) must not be a silent alert miss.
    resp = MagicMock()
    resp.status_code = 401
    resp.text = "Unauthorized"
    with patch.object(run_perp_shadow.requests, "post", return_value=resp), \
            caplog.at_level("WARNING"):
        ok = run_perp_shadow.send_telegram("tok", "chat", "hi")
    assert ok is False
    assert "Telegram send failed: HTTP 401" in caplog.text
