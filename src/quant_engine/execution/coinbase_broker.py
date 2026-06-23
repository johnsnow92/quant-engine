"""Coinbase Advanced Trade live execution broker.

Routes orders to Coinbase INTX perpetuals via direct REST.
Sits behind the same PreTradeGuard as PaperBroker — every order
passes the guard before any network call is made.

Required env vars (load from Infisical before running):
    COINBASE_CDP_API_KEY_NAME   e.g. organizations/{org}/apiKeys/{id}
    COINBASE_CDP_PRIVATE_KEY    EC private key PEM (newlines as \\n)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field

from coinbase.rest import RESTClient

from .guards import GuardRejection, Order, PreTradeGuard
from .paper_broker import Fill

log = logging.getLogger(__name__)

# Internal instrument name → Coinbase INTX product_id
_INSTRUMENT_MAP: dict[str, str] = {
    "BTCUSD-PERP": "BTC-PERP-INTX",
    "ETHUSD-PERP": "ETH-PERP-INTX",
}

_TERMINAL_STATUSES = {"FILLED", "CANCELLED", "FAILED", "EXPIRED"}
_POLL_INTERVAL = 0.3   # seconds between status polls
_POLL_ATTEMPTS = 10
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 1.0   # seconds


class ExecutionError(Exception):
    def __init__(self, message: str, non_retriable: bool = False):
        super().__init__(message)
        self.non_retriable = non_retriable


@dataclass
class CoinbaseBroker:
    """Live broker backed by Coinbase Advanced Trade INTX perpetuals.

    Implements the same duck-type as PaperBroker so the execution layer
    is swappable without changing strategy or signal code.
    """
    guard: PreTradeGuard
    dry_run: bool = False
    _client: RESTClient = field(init=False, repr=False)
    _positions: dict[str, float] = field(default_factory=dict, init=False)
    _position_cache_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        key_name = os.environ.get("COINBASE_CDP_API_KEY_NAME", "")
        private_key = os.environ.get("COINBASE_CDP_PRIVATE_KEY", "").replace("\\n", "\n")
        if not key_name or not private_key:
            raise EnvironmentError(
                "COINBASE_CDP_API_KEY_NAME and COINBASE_CDP_PRIVATE_KEY must be set"
            )
        self._client = RESTClient(api_key=key_name, api_secret=private_key)

    # ------------------------------------------------------------------
    # Public interface (mirrors PaperBroker)
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> Fill:
        """Validate, route, and fill one order. Raises on any failure."""
        current = self._get_position(order.instrument)
        self.guard.check(order, current_position=current)

        product_id = _INSTRUMENT_MAP.get(order.instrument)
        if product_id is None:
            raise GuardRejection(
                f"instrument {order.instrument!r} has no Coinbase product_id mapping"
            )

        if self.dry_run:
            log.info("DRY RUN — would submit %s %s qty=%.6f @ ~%.2f",
                     order.side, order.instrument, order.qty, order.price)
            fill = Fill(order.instrument, order.side, order.qty, order.price, fee=0.0)
            self._positions[order.instrument] = current + order.signed_qty
            return fill

        fill = self._submit_with_retry(order, product_id, current)
        return fill

    def position(self, instrument: str) -> float:
        return self._get_position(instrument)

    def equity(self, marks: dict[str, float]) -> float:
        """Cash-equivalent equity: sum of mark-to-market position values."""
        self._ensure_position_cache()
        return sum(
            qty * marks.get(inst, 0.0)
            for inst, qty in self._positions.items()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_position(self, instrument: str) -> float:
        self._ensure_position_cache()
        return self._positions.get(instrument, 0.0)

    def _ensure_position_cache(self) -> None:
        if self._position_cache_loaded:
            return
        try:
            portfolios = self._client.get_portfolios()
            for portfolio in (portfolios.get("portfolios") or []):
                breakdown = self._client.get_portfolio_breakdown(
                    portfolio_uuid=portfolio["uuid"]
                )
                for pos in (breakdown.get("breakdown", {}).get("perp_positions") or []):
                    internal = _reverse_map(pos.get("product_id", ""))
                    if internal:
                        net = float(pos.get("net_size") or 0)
                        self._positions[internal] = net
            self._position_cache_loaded = True
        except Exception as exc:
            log.warning("Could not load position cache from Coinbase: %s — assuming flat", exc)
            self._position_cache_loaded = True

    def _submit_with_retry(self, order: Order, product_id: str, current: float) -> Fill:
        client_order_id = str(uuid.uuid4())
        last_exc: Exception | None = None

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return self._submit_once(order, product_id, client_order_id, current)
            except ExecutionError as exc:
                if exc.non_retriable:
                    raise
                last_exc = exc
                log.warning("Order attempt %d/%d failed: %s", attempt, _RETRY_ATTEMPTS, exc)
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF)
            except Exception as exc:
                last_exc = exc
                log.warning("Order attempt %d/%d error: %s", attempt, _RETRY_ATTEMPTS, exc)
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF)

        raise ExecutionError(f"Order failed after {_RETRY_ATTEMPTS} attempts: {last_exc}")

    def _submit_once(
        self, order: Order, product_id: str, client_order_id: str, current: float
    ) -> Fill:
        response = self._client.market_order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=order.side.upper(),
            base_size=str(order.qty),
        )

        success = response.get("success", False)
        if not success:
            err = response.get("error_response") or response
            status_code = response.get("preview_failure_reason", "")
            non_retriable = "insufficient" in str(err).lower() or "invalid" in str(err).lower()
            raise ExecutionError(f"Order rejected by Coinbase: {err}", non_retriable=non_retriable)

        order_id = (
            response.get("success_response", {}).get("order_id")
            or response.get("order_id", "")
        )
        if not order_id:
            raise ExecutionError("Coinbase returned no order_id", non_retriable=True)

        fill = self._poll_until_filled(order_id, order)
        self._positions[order.instrument] = current + order.signed_qty
        return fill

    def _poll_until_filled(self, order_id: str, order: Order) -> Fill:
        for _ in range(_POLL_ATTEMPTS):
            time.sleep(_POLL_INTERVAL)
            try:
                detail = self._client.get_order(order_id=order_id)
                o = detail.get("order") or detail
                status = o.get("status", "")
                if status not in _TERMINAL_STATUSES:
                    continue
                if status != "FILLED":
                    raise ExecutionError(
                        f"Order {order_id} ended with status {status}", non_retriable=True
                    )
                avg_price = float(o.get("average_filled_price") or order.price)
                fee = float(o.get("total_fees") or 0.0)
                return Fill(order.instrument, order.side, order.qty, avg_price, fee)
            except ExecutionError:
                raise
            except Exception as exc:
                log.debug("Poll error for %s: %s", order_id, exc)

        raise ExecutionError(
            f"Order {order_id} did not reach terminal status after {_POLL_ATTEMPTS} polls"
        )


def _reverse_map(product_id: str) -> str | None:
    """Coinbase product_id → internal instrument name, or None if unmapped."""
    for internal, cb_id in _INSTRUMENT_MAP.items():
        if cb_id == product_id:
            return internal
    return None
