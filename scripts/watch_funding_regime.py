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
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
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
    if state.is_on and prev_on is False:
        header = "🟢 *Quant-engine regime TURNED ON*"
    elif state.is_on and prev_on is None:
        header = "🟢 *Quant-engine regime is ON*"
    else:
        header = "🟢 *Quant-engine regime ON — carry conditions active*"

    lines = [
        header,
        "",
        f"```",
        state.summary(),
        f"```",
        "",
        "Action: deploy capital to quant-engine carry strategy or review gates.",
    ]
    return "\n".join(lines)


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
        )
    except Exception as exc:
        log.error("Failed to fetch regime state: %s", exc)
        sys.exit(2)

    print(state.summary())

    prev_on = load_last_state(args.state_file) if args.state_file else None
    if args.state_file:
        save_state(args.state_file, state)

    should_alert = False
    if state.is_on:
        if args.always_alert:
            should_alert = True
        elif args.alert_on_change_only:
            should_alert = (prev_on is not True)  # alert only on OFF→ON transition
        else:
            should_alert = True  # default: alert whenever ON

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
