"""Funding-regime watcher — the quant-engine wake trigger.

Fetches live funding rates from Crypto.com and Binance (via CoinDesk), evaluates
whether conditions are favourable for the carry strategy, and fires a Telegram
alert when the regime is ON.

Exits 0  = regime OFF  (GitHub Actions treats this as success, no further steps)
Exits 1  = regime ON   (workflow can gate downstream jobs on this exit code)

Usage (local):
    uv run python scripts/watch_funding_regime.py

Usage (CI / GitHub Actions — see .github/workflows/funding-regime-watcher.yml):
    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... uv run python scripts/watch_funding_regime.py \\
        --state-file /tmp/regime_state.json --alert-on-change-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests

from quant_engine.watchers.funding_regime import RegimeState, check_regime

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except requests.HTTPError as exc:
        # Surface Telegram's own error description ("chat not found",
        # "can't parse entities", ...) but NEVER log the raw exception — its
        # string embeds the request URL, which contains the bot token.
        resp = exc.response
        status = resp.status_code if resp is not None else "?"
        reason = resp.reason if resp is not None else ""
        detail = ""
        if resp is not None:
            try:
                detail = resp.json().get("description", "")
            except ValueError:
                detail = resp.text[:200]
        log.warning("Telegram send failed: HTTP %s %s — %s", status, reason, detail)
        return False
    except Exception as exc:
        # Connection/timeout errors can also embed the tokenized URL in their
        # message, so log only the exception type, never str(exc).
        log.warning("Telegram send failed: %s", type(exc).__name__)
        return False


def load_last_state(path: Path) -> bool | None:
    """Return last known regime ON/OFF, or None if no state file."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("is_on")
    except Exception:
        return None


def save_state(path: Path, state: RegimeState) -> None:
    path.write_text(json.dumps({
        "is_on": state.is_on,
        "checked_at": state.checked_at.isoformat(),
        "triggered_by": state.triggered_by,
    }))


def build_alert(state: RegimeState, prev_on: bool | None) -> str:
    deploy_action = "Action: deploy capital to quant-engine carry strategy or review gates."
    if not state.is_on:
        header = "⚪️ *Quant-engine regime OFF — carry conditions not met*"
        action = "Action: none — forced/heartbeat alert confirming the watcher is live."
    elif prev_on is False:
        header = "🟢 *Quant-engine regime TURNED ON*"
        action = deploy_action
    elif prev_on is None:
        header = "🟢 *Quant-engine regime is ON*"
        action = deploy_action
    else:
        header = "🟢 *Quant-engine regime ON — carry conditions active*"
        action = deploy_action

    lines = [
        header,
        "",
        "```",
        state.summary(),
        "```",
        "",
        action,
    ]
    return "\n".join(lines)


def should_send_alert(
    is_on: bool,
    *,
    always_alert: bool,
    alert_on_change_only: bool,
    prev_on: bool | None,
) -> bool:
    """Decide whether to fire a Telegram alert this cycle.

    --always-alert fires regardless of regime state (manual wiring test or
    daily digest). Otherwise only ON states alert: on-change-only fires on the
    OFF->ON transition or when the prior state is unknown (no state file yet);
    the default fires whenever the regime is ON.
    """
    if always_alert:
        return True
    if not is_on:
        return False
    if alert_on_change_only:
        return prev_on is not True  # only on OFF->ON transition
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant-engine funding-regime watcher")
    parser.add_argument(
        "--single-venue-threshold",
        type=float,
        default=0.15,
        help="Min annualized single-venue funding to trigger (default 15%%)",
    )
    parser.add_argument(
        "--spread-threshold",
        type=float,
        default=0.05,
        help="Min annualized cross-venue spread to trigger (default 5%%)",
    )
    parser.add_argument(
        "--carry-threshold",
        type=float,
        default=0.05,
        help="Min trailing-7d BTC perp funding, annualized, for the carry regime (default 5%%)",
    )
    parser.add_argument(
        "--carry-lookback",
        type=int,
        default=168,
        help="Bars (1h) for the carry-regime window (default 168 = 7 days)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="JSON file to persist last regime state for transition detection",
    )
    parser.add_argument(
        "--alert-on-change-only",
        action="store_true",
        help="Only send Telegram when regime transitions ON (not when already ON)",
    )
    parser.add_argument(
        "--always-alert",
        action="store_true",
        help="Send Telegram regardless of state (useful for daily digests)",
    )
    args = parser.parse_args()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    log.info("Checking funding regime...")
    try:
        state = check_regime(
            single_venue_threshold=args.single_venue_threshold,
            spread_threshold=args.spread_threshold,
            carry_threshold=args.carry_threshold,
            carry_lookback=args.carry_lookback,
        )
    except Exception as exc:
        log.error("Failed to fetch regime state: %s", exc)
        sys.exit(2)

    print(state.summary())

    prev_on = load_last_state(args.state_file) if args.state_file else None
    if args.state_file:
        save_state(args.state_file, state)

    should_alert = should_send_alert(
        state.is_on,
        always_alert=args.always_alert,
        alert_on_change_only=args.alert_on_change_only,
        prev_on=prev_on,
    )

    if should_alert:
        if telegram_token and telegram_chat_id:
            msg = build_alert(state, prev_on)
            sent = send_telegram(telegram_token, telegram_chat_id, msg)
            log.info("Telegram alert %s", "sent" if sent else "FAILED")
        else:
            log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping alert")

    sys.exit(1 if state.is_on else 0)


if __name__ == "__main__":
    main()
