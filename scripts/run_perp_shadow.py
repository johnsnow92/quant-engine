"""Perp dead-band carry — SHADOW loop runner (spec docs/plans/07 §4.6).

Wires the merged shadow machinery into a runnable entrypoint: build the two
read-only adapters from env credentials, run one ``PerpShadowRunner`` cycle, log
the decision, and optionally alert it to Telegram.

SHADOW ONLY. The runner places nothing — PaperBroker sits on both legs and the
live adapters are read-only (no ``submit_order``). There is no live mode, no MODE
flag, and no order path in this script.

Graceful by design for an unattended cron:
  * Missing perp read credentials → log "blocked" and exit 0. Not-yet-provisioned
    is the expected default state, not a failure.
  * Any live-read error (the adapter endpoints are confirm-at-micro, spec §11) →
    log and exit 0; a data hiccup must never fail the scheduled job.

Env:
  KALSHI_PERP_KEY_ID, KALSHI_PERP_PRIVATE_KEY          Kalshi perp read creds
  COINBASE_CFM_API_KEY_NAME, COINBASE_CFM_PRIVATE_KEY  Coinbase CFM read creds
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID                 optional alerting
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("perp_shadow")

_KALSHI_ENV = ("KALSHI_PERP_KEY_ID", "KALSHI_PERP_PRIVATE_KEY")
_CFM_ENV = ("COINBASE_CFM_API_KEY_NAME", "COINBASE_CFM_PRIVATE_KEY")
_TELEGRAM_TIMEOUT = 15


def missing_credentials() -> list[str]:
    """Names of the required read-credential env vars that are unset."""
    return [name for name in (*_KALSHI_ENV, *_CFM_ENV) if not os.getenv(name)]


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url, json={"chat_id": chat_id, "text": text}, timeout=_TELEGRAM_TIMEOUT
        )
    except requests.RequestException as exc:
        log.warning("Telegram send failed: %s", exc)
        return False
    if resp.status_code != 200:
        # Non-200 (bad token/chat_id, rate limit) would otherwise be a silent miss.
        log.warning("Telegram send failed: HTTP %s %s", resp.status_code, resp.text[:200])
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Perp carry SHADOW loop (places nothing)")
    parser.add_argument("--instrument", default="BTCUSD-PERP")
    parser.add_argument(
        "--always-alert",
        action="store_true",
        help="Telegram every cycle (default: only on executed/blocked decisions)",
    )
    args = parser.parse_args()

    missing = missing_credentials()
    if missing:
        log.info(
            "perp shadow BLOCKED — missing read credentials: %s. Set the "
            "KALSHI_PERP_* and COINBASE_CFM_* secrets to activate; until then the "
            "shadow loop cannot read live funding/marks.",
            ", ".join(missing),
        )
        return 0

    from quant_engine.execution.coinbase_cfm_broker import CoinbaseCfmReader
    from quant_engine.execution.kalshi_perp_broker import KalshiPerpReader
    from quant_engine.execution.shadow_runner import (
        BLOCKED_BY_GATES,
        EXECUTED_SHADOW,
        PerpShadowRunner,
    )

    try:
        runner = PerpShadowRunner(
            long_reader=KalshiPerpReader(),
            short_reader=CoinbaseCfmReader(),
            instrument=args.instrument,
        )
        decision = runner.run_cycle()
    except Exception as exc:  # noqa: BLE001 — a read/endpoint hiccup must not fail the cron
        log.warning(
            "perp shadow read/cycle failed (adapter endpoints are confirm-at-micro): %s",
            exc,
            exc_info=True,
        )
        return 0

    summary = decision.summary()
    log.info(summary)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    notable = decision.action in (EXECUTED_SHADOW, BLOCKED_BY_GATES)
    if token and chat_id and (args.always_alert or notable):
        send_telegram(token, chat_id, f"[perp shadow] {summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
