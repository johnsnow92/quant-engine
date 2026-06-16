"""Kalshi perp read-only adapter — spec docs/plans/07 §4.1 (the long, 0%-funding leg).

The long leg of the perp dead-band carry sits on Kalshi's CFTC-regulated perp
margin surface, whose funding pins at ≈0% in the dead band (confirmed live:
portfolio read 2026-06-13 showed Fund 0.0000%). This adapter is READ-ONLY:
positions, margin buffer, mark price, funding rate — each normalized into the
type the deterministic safety stack already consumes (`reconcile.LegSnapshot`,
the edge-clears-fees gate input, the annualized rate the funding-convention
check reconciles). No order placement here; orders route through the atomic
two-leg executor.

Auth is Kalshi's RSA-PSS scheme: each request is signed over
``timestamp_ms + METHOD + path`` with the perp key (distinct from the
predictions key, per Kalshi's per-surface key rule). `sign_pss` is a pure,
unit-tested function; the live HTTP transport is an injectable
``requests.Session`` (house style), so the whole adapter tests against mocks.

Env (Infisical only):
    KALSHI_PERP_KEY_ID         API key id (UUID) for the perp/margin surface
    KALSHI_PERP_PRIVATE_KEY    RSA private key PEM (newlines as \\n)

The base host, endpoint paths, BTC-per-contract size, and funding interval are
configurable and must be confirmed against the live perp production API at the
micro stage (spec §4.6, open question §11 — perp prod access verified via the
read-only ``/margin/enabled`` endpoint). The funding-convention check (§4.5) and
the reconciliation daemon (§4.4) backstop any normalization mismatch before
capital scales. `cryptography` is required for signing (present transitively via
coinbase-advanced-py; promote to a direct dependency when convenient).
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field

import requests

from .reconcile import LegSnapshot

log = logging.getLogger(__name__)

VENUE = "kalshi-perp"
_HOURS_PER_YEAR = 8_760.0
DEFAULT_TIMEOUT = 20

# Confirm host + prefix against the perp production API at micro (spec §11).
_DEFAULT_BASE_URL = "https://external-api.kalshi.com"
_API_PREFIX = "/trade-api/v2"

# Read-only endpoint paths (relative to the API prefix). Confirm perp/margin
# paths at micro — these are the documented predictions-surface shapes.
_PATH_POSITIONS = "/portfolio/positions"
_PATH_MARGIN = "/margin/summary"
_PATH_MARKET = "/markets/{ticker}"


@dataclass(frozen=True)
class KalshiPerpProduct:
    """Maps an internal instrument to its Kalshi perp ticker + contract size."""
    ticker: str
    btc_per_contract: float   # BTC exposure per Kalshi perp contract; confirm at micro
    funding_interval_hours: float


# Internal instrument → Kalshi perp product. Confirm ticker + size at micro.
_KALSHI_PERP_PRODUCTS: dict[str, KalshiPerpProduct] = {
    "BTCUSD-PERP": KalshiPerpProduct(ticker="BTCUSD-PERP", btc_per_contract=0.001, funding_interval_hours=1.0),
    "ETHUSD-PERP": KalshiPerpProduct(ticker="ETHUSD-PERP", btc_per_contract=0.01, funding_interval_hours=1.0),
}


class KalshiDataError(Exception):
    """Raised when a Kalshi response can't be parsed into a safety-stack type."""


def signed_message(timestamp_ms: int, method: str, path: str) -> str:
    """The exact string Kalshi signs: timestamp + METHOD + path (no separators)."""
    return f"{timestamp_ms}{method.upper()}{path}"


def sign_pss(private_key_pem: str, message: str) -> str:
    """RSA-PSS / SHA-256 signature, base64-encoded — Kalshi's auth scheme.

    Salt length is the digest length (Kalshi's documented requirement). Pure:
    given a key and message it always returns the same-shape signature, so it is
    verified in tests with a generated keypair without any network.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    signature = key.sign(
        message.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


@dataclass
class KalshiPerpReader:
    """Read-only view of the Kalshi perp/margin account.

    Duck-typed for the reconciliation daemon: `leg_snapshot()` returns the
    signed position + margin buffer the reconciler asserts every cycle.
    """
    base_url: str = _DEFAULT_BASE_URL
    session: requests.Session = field(default_factory=requests.Session)
    _key_id: str = field(init=False, repr=False)
    _private_key_pem: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._key_id = os.environ.get("KALSHI_PERP_KEY_ID", "")
        self._private_key_pem = os.environ.get("KALSHI_PERP_PRIVATE_KEY", "").replace("\\n", "\n")
        if not self._key_id or not self._private_key_pem:
            raise EnvironmentError(
                "KALSHI_PERP_KEY_ID and KALSHI_PERP_PRIVATE_KEY must be set"
            )

    # ------------------------------------------------------------------
    # Read-only reads, each normalized to a safety-stack input
    # ------------------------------------------------------------------

    def product(self, instrument: str) -> KalshiPerpProduct:
        prod = _KALSHI_PERP_PRODUCTS.get(instrument)
        if prod is None:
            raise KalshiDataError(f"instrument {instrument!r} has no Kalshi perp mapping")
        return prod

    def position_btc(self, instrument: str) -> float:
        """Signed base-unit position (+ long / - short). Absent ticker = flat."""
        prod = self.product(instrument)
        data = self._get(_PATH_POSITIONS)
        if "market_positions" not in data:
            raise KalshiDataError(
                "Kalshi positions payload missing 'market_positions' — refusing to report flat"
            )
        for mp in data["market_positions"] or []:
            if mp.get("ticker") == prod.ticker:
                raw = mp.get("position") or 0
                try:
                    contracts = float(raw)
                except (TypeError, ValueError) as exc:
                    raise KalshiDataError(f"Kalshi position not numeric: {raw!r}") from exc
                return contracts * prod.btc_per_contract
        return 0.0

    def margin_buffer_pct(self) -> float:
        """Percent the account equity sits above its maintenance margin."""
        data = self._get(_PATH_MARGIN)
        summary = data.get("margin") or data
        return _kalshi_margin_buffer_pct(summary)

    def mark_price(self, instrument: str) -> float:
        prod = self.product(instrument)
        data = self._get(_PATH_MARKET.format(ticker=prod.ticker))
        market = data.get("market") or data
        return _kalshi_mark_price(market)

    def funding_rate_annual(self, instrument: str) -> float:
        """Annualized funding (signed) from the venue's per-interval rate.

        Pinned at ≈0 in the dead band — the long leg's whole appeal. Annualized
        here so the safety stack reasons in one unit; the funding-convention
        check guards the interval conversion.
        """
        prod = self.product(instrument)
        data = self._get(_PATH_MARKET.format(ticker=prod.ticker))
        market = data.get("market") or data
        rate = market.get("funding_rate")
        if rate is None:
            raise KalshiDataError(f"no funding rate for {prod.ticker}")
        try:
            rate_f = float(rate)
        except (TypeError, ValueError) as exc:
            raise KalshiDataError(f"Kalshi funding_rate not numeric: {rate!r}") from exc
        intervals_per_year = _HOURS_PER_YEAR / prod.funding_interval_hours
        return rate_f * intervals_per_year

    def leg_snapshot(self, instrument: str) -> LegSnapshot:
        """The reconciler's per-cycle input for the long (Kalshi) leg."""
        return LegSnapshot(
            venue=VENUE,
            position_btc=self.position_btc(instrument),
            margin_buffer_pct=self.margin_buffer_pct(),
        )

    # ------------------------------------------------------------------
    # Signed transport
    # ------------------------------------------------------------------

    def _auth_headers(self, method: str, signed_path: str, timestamp_ms: int) -> dict[str, str]:
        signature = sign_pss(self._private_key_pem, signed_message(timestamp_ms, method, signed_path))
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

    def _get(self, endpoint: str) -> dict:
        signed_path = _API_PREFIX + endpoint
        timestamp_ms = int(time.time() * 1000)
        headers = self._auth_headers("GET", signed_path, timestamp_ms)
        resp = self.session.get(self.base_url + signed_path, headers=headers, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()


# ----------------------------------------------------------------------
# Pure normalization helpers (the testable core)
# ----------------------------------------------------------------------

def _kalshi_margin_buffer_pct(summary: dict) -> float:
    """Percent above maintenance from a Kalshi margin summary.

    Prefers an explicit buffer percent if supplied; otherwise derives it from
    portfolio value vs maintenance margin. Amount fields may be USD or cents
    objects — only ratios are taken, so the unit cancels.
    """
    explicit = summary.get("margin_buffer_percentage")
    if explicit is not None:
        return float(explicit)

    equity = _kalshi_amount(summary, "portfolio_value", "available_balance", "equity")
    maintenance = _kalshi_amount(summary, "maintenance_margin", "required_margin")
    if equity is None or maintenance is None or maintenance <= 0:
        raise KalshiDataError("cannot derive margin buffer from Kalshi margin summary")
    return (equity - maintenance) / maintenance * 100.0


def _kalshi_mark_price(market: dict) -> float:
    """Mark price for the perp: explicit mark, else bid/ask mid."""
    mark = market.get("mark_price") or market.get("last_price")
    if mark is not None:
        return float(mark)
    bid = market.get("yes_bid")
    ask = market.get("yes_ask")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    raise KalshiDataError("no mark price on Kalshi market payload")


def _kalshi_amount(summary: dict, *keys: str) -> float | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if value is not None:
            return float(value)
    return None
