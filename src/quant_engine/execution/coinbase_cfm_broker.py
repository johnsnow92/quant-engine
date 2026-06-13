"""Coinbase Financial Markets (CFM) read-only adapter — spec docs/plans/07 §4.1.

The short leg of the perp dead-band carry lives on Coinbase's US-legal,
CFTC-regulated futures (CFM / Nodal) — NOT spot, and NOT the international INTX
perps (those are off-allowlist). This adapter is deliberately READ-ONLY: it
pulls positions, the margin buffer, mark price, and the funding rate, and
normalizes each into a type the deterministic safety stack already consumes —
`reconcile.LegSnapshot`, the edge-clears-fees gate input, and the annualized
rate the funding-convention check (`funding_check`) reconciles against realized
cash. Order placement is intentionally absent; orders route through the atomic
two-leg executor, never from a data adapter.

The connected `coinbase` CLI reads spot only, so this talks to the CFM-scoped
Advanced Trade API via a distinct key (perp surface ≠ spot surface).

Env (Infisical only):
    COINBASE_CFM_API_KEY_NAME   organizations/{org}/apiKeys/{id}
    COINBASE_CFM_PRIVATE_KEY    EC private key PEM (newlines as \\n)

The exact CFM product IDs, contract sizes, and funding-field names are
configurable and must be confirmed against live CFM responses at the micro
stage (spec §4.6). The funding-convention check (§4.5) and the reconciliation
daemon (§4.4) are the backstops that catch any normalization mismatch before
capital scales.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from coinbase.rest import RESTClient

from .reconcile import LegSnapshot

log = logging.getLogger(__name__)

VENUE = "coinbase-futures"
_HOURS_PER_YEAR = 8_760.0


@dataclass(frozen=True)
class CfmProduct:
    """Maps an internal instrument to its CFM product and venue conventions."""
    product_id: str
    contract_size_base: float       # base units (e.g. BTC) per contract; nano BTC = 0.01
    funding_interval_hours: float   # CFM perps fund hourly


# Internal instrument → CFM product. Confirm product_id + contract size at micro.
_CFM_PRODUCTS: dict[str, CfmProduct] = {
    "BTCUSD-PERP": CfmProduct(product_id="BTC-PERP", contract_size_base=0.01, funding_interval_hours=1.0),
    "ETHUSD-PERP": CfmProduct(product_id="ETH-PERP", contract_size_base=0.10, funding_interval_hours=1.0),
}


class CfmDataError(Exception):
    """Raised when a CFM response can't be parsed into a safety-stack type."""


@dataclass
class CoinbaseCfmReader:
    """Read-only view of the Coinbase CFM futures account.

    Duck-typed for the reconciliation daemon: `leg_snapshot()` returns the
    signed position + margin buffer the reconciler asserts on every cycle.
    """
    _client: RESTClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key_name = os.environ.get("COINBASE_CFM_API_KEY_NAME", "")
        private_key = os.environ.get("COINBASE_CFM_PRIVATE_KEY", "").replace("\\n", "\n")
        if not key_name or not private_key:
            raise EnvironmentError(
                "COINBASE_CFM_API_KEY_NAME and COINBASE_CFM_PRIVATE_KEY must be set"
            )
        self._client = RESTClient(api_key=key_name, api_secret=private_key)

    # ------------------------------------------------------------------
    # Read-only reads, each normalized to a safety-stack input
    # ------------------------------------------------------------------

    def product(self, instrument: str) -> CfmProduct:
        prod = _CFM_PRODUCTS.get(instrument)
        if prod is None:
            raise CfmDataError(f"instrument {instrument!r} has no CFM product mapping")
        return prod

    def position_btc(self, instrument: str) -> float:
        """Signed base-unit position (+ long / - short) for one CFM product."""
        prod = self.product(instrument)
        raw = self._client.get_futures_position(product_id=prod.product_id)
        pos = raw.get("position") or raw
        return _signed_size(pos, prod.contract_size_base)

    def margin_buffer_pct(self) -> float:
        """Percent the account equity sits above its liquidation threshold."""
        raw = self._client.get_futures_balance_summary()
        summary = raw.get("balance_summary") or raw
        return _margin_buffer_pct(summary)

    def mark_price(self, instrument: str) -> float:
        prod = self.product(instrument)
        product = _unwrap_product(self._client.get_product(product_id=prod.product_id))
        price = product.get("price") or product.get("mark_price")
        if price is None:
            raise CfmDataError(f"no mark price for {prod.product_id}")
        return float(price)

    def funding_rate_annual(self, instrument: str) -> float:
        """Annualized funding rate (signed) from the venue's per-interval rate.

        CFM perps post funding hourly; the safety stack reasons in annual terms,
        so the adapter does the interval→annual conversion here. This is exactly
        the conversion the funding-convention check guards against getting wrong
        (the 8h-vs-hourly class of bug), which is why the two are paired.
        """
        prod = self.product(instrument)
        product = _unwrap_product(self._client.get_product(product_id=prod.product_id))
        details = (
            product.get("future_product_details")
            or product.get("perpetual_details")
            or product
        )
        rate = details.get("funding_rate")
        if rate is None:
            raise CfmDataError(f"no funding rate for {prod.product_id}")
        intervals_per_year = _HOURS_PER_YEAR / prod.funding_interval_hours
        return float(rate) * intervals_per_year

    def leg_snapshot(self, instrument: str) -> LegSnapshot:
        """The reconciler's per-cycle input for the short (CFM) leg."""
        return LegSnapshot(
            venue=VENUE,
            position_btc=self.position_btc(instrument),
            margin_buffer_pct=self.margin_buffer_pct(),
        )


# ----------------------------------------------------------------------
# Pure normalization helpers (the testable core)
# ----------------------------------------------------------------------

def _unwrap_product(raw: dict) -> dict:
    return raw.get("product") or raw


def _signed_size(pos: dict, contract_size_base: float) -> float:
    """Normalize a CFM position payload to a signed base-unit quantity.

    Handles both documented shapes: an already-base-denominated signed
    ``net_size``, or ``number_of_contracts`` + ``side`` that must be multiplied
    by the contract size and signed. Raises if neither is present rather than
    silently returning 0 (a flat reading on a live position is a safety hole).
    """
    net = pos.get("net_size")
    if net is not None:
        return float(net)
    contracts = pos.get("number_of_contracts")
    if contracts is None:
        raise CfmDataError("CFM position has neither net_size nor number_of_contracts")
    side = str(pos.get("side", "")).upper()
    sign = -1.0 if side == "SHORT" else 1.0
    return sign * float(contracts) * contract_size_base


def _margin_buffer_pct(summary: dict) -> float:
    """Percent above liquidation from a CFM balance summary.

    Prefers an explicit ``liquidation_buffer_percentage`` if the venue supplies
    it; otherwise derives it from equity vs the maintenance/liquidation
    threshold. Amount fields may be plain numbers or ``{"value": ...}`` objects.
    """
    explicit = summary.get("liquidation_buffer_percentage")
    if explicit is not None:
        return float(explicit)

    equity = _amount(summary, "total_balance", "cfm_usd_balance")
    maintenance = _amount(summary, "liquidation_threshold", "maintenance_margin")
    if equity is None or maintenance is None or maintenance <= 0:
        raise CfmDataError("cannot derive margin buffer from CFM balance summary")
    return (equity - maintenance) / maintenance * 100.0


def _amount(summary: dict, *keys: str) -> float | None:
    """First parseable USD amount among ``keys`` (plain or ``{value}``-wrapped)."""
    for key in keys:
        value = summary.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if value is not None:
            return float(value)
    return None
