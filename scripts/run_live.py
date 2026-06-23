"""Live carry execution entry point.

Checks the funding regime, builds a carry signal, and submits one order
through the CoinbaseBroker if the regime is ON.

Usage:
    uv run python scripts/run_live.py [--dry-run]

    --dry-run   Guard check + broker construction + signal generation, but no
                real order submitted. Exit 0 on success. Used in CI.

Secrets (load from Infisical before running):
    COINBASE_CDP_API_KEY_NAME
    COINBASE_CDP_PRIVATE_KEY
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (optional — for Telegram alerts)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("run_live")


def _carry_signal(regime):
    """Derive a single carry Order from the live RegimeState.

    Uses BTC Crypto.com annualized funding as the signal:
    - positive funding → short perp (sell), long spot implied
    - negative funding → skip (we never go short the basis at current thresholds)

    Returns Order or None.
    """
    from quant_engine.execution.guards import Order

    # Only trade BTC perp for Tranche 1
    ann = regime.btc_cc_ann
    if ann <= 0:
        log.info("BTC funding not positive (%.1%%) — no trade", ann)
        return None

    # Size: fixed 0.01 BTC starter size for Tranche 1
    qty = 0.01
    price = 1.0  # market order — price is a dummy for notional guard check
    # Fetch a real mark price so the guard notional check is meaningful
    try:
        from quant_engine.data.cryptocom import CryptoComClient
        cc = CryptoComClient()
        ticker = cc.fetch_ticker("BTCUSD-PERP")
        price = float(ticker.get("mark_price") or ticker.get("last_price") or 60_000)
    except Exception as exc:
        log.warning("Could not fetch mark price: %s — using fallback 60000", exc)
        price = 60_000.0

    return Order(instrument="BTCUSD-PERP", side="sell", qty=qty, price=price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant-engine live carry executor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate path without submitting a real order")
    args = parser.parse_args()

    # Load secrets from ~/.claude/.env or project .env if not already in env
    for env_path in [
        os.path.expanduser("~/.claude/.env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]:
        if os.path.exists(env_path):
            with open(env_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v

    from quant_engine.watchers.funding_regime import check_regime
    from quant_engine.execution.guards import PreTradeGuard
    from quant_engine.execution.coinbase_broker import CoinbaseBroker
    from quant_engine.execution.guards import GuardRejection
    from quant_engine.execution.coinbase_broker import ExecutionError

    log.info("Checking funding regime...")
    try:
        regime = check_regime()
    except Exception as exc:
        log.error("Regime check failed: %s", exc)
        sys.exit(2)

    log.info(regime.summary())

    if not regime.is_on:
        log.info("Regime is OFF — no execution this cycle")
        sys.exit(0)

    guard = PreTradeGuard(
        allowed_instruments={"BTCUSD-PERP", "ETHUSD-PERP"},
        max_notional_usd=2_000.0,
        max_position_qty=0.025,
    )

    try:
        broker = CoinbaseBroker(guard=guard, dry_run=args.dry_run)
    except EnvironmentError as exc:
        log.error("Broker init failed: %s", exc)
        sys.exit(2)

    signal = _carry_signal(regime)
    if signal is None:
        log.info("No carry signal — exiting")
        sys.exit(0)

    log.info("Signal: %s %s qty=%.4f @ ~%.2f", signal.side, signal.instrument,
             signal.qty, signal.price)

    try:
        fill = broker.submit_order(signal)
        log.info("Fill: %s %s qty=%.4f @ %.4f fee=%.4f",
                 fill.side, fill.instrument, fill.qty, fill.price, fill.fee)
    except GuardRejection as exc:
        log.warning("Guard rejected order: %s", exc)
        sys.exit(0)
    except ExecutionError as exc:
        log.error("Execution failed: %s", exc)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
