"""Funding-convention verification — spec docs/plans/07 §4.5. THE #1 silent-P&L risk.

On the first N round-trips we reconcile the EXPECTED funding (each venue's stated
interval / sign / rate applied to the position over the elapsed time) against the
REALIZED cash that actually posted. A mismatch means we modelled funding wrong —
wrong sign (direction), or wrong magnitude (8h vs hourly interval, or rate- vs
price-denominated, the Kraken normalization quirk the funding logger caught). Halt
and alert before scaling; the carry P&L is not trustworthy until this passes.

Pure + deterministic. Funding convention: when the rate is positive, LONGS PAY
SHORTS — so a long position loses cash, a short receives.
"""
from __future__ import annotations

from dataclasses import dataclass

_HOURS_PER_YEAR = 8_760.0


@dataclass(frozen=True)
class FundingObservation:
    venue: str
    position_btc: float           # signed: + long, - short
    mark_price_usd: float
    stated_rate_annual: float     # venue's stated funding rate, annualized, signed
    elapsed_hours: float
    realized_funding_usd: float   # cash that actually posted over elapsed_hours


def expected_funding_usd(obs: FundingObservation) -> float:
    """Expected funding cash from the stated convention.

    notional * rate * (elapsed / year), signed so a long pays (negative cash) and a
    short receives (positive) when the rate is positive. The position sign and the
    rate sign both flow through, so a negative rate flips it correctly.
    """
    notional = abs(obs.position_btc) * obs.mark_price_usd
    direction = -1.0 if obs.position_btc > 0 else 1.0   # long pays, short receives
    return direction * notional * obs.stated_rate_annual * (obs.elapsed_hours / _HOURS_PER_YEAR)


def verify_funding(
    obs: FundingObservation, tolerance_usd: float = 1.0, rel_tolerance: float = 0.10
) -> tuple[bool, str]:
    """True iff realized funding matches the stated convention within tolerance.

    Catches direction (sign) errors and magnitude errors (which is how an interval or
    denomination mistake shows up). Passes within an absolute floor OR a relative band
    so tiny micro-size amounts (dominated by noise) don't false-positive.
    """
    expected = expected_funding_usd(obs)
    realized = obs.realized_funding_usd

    if expected != 0.0 and realized * expected < 0.0:
        return False, (
            f"{obs.venue} funding SIGN mismatch: expected {expected:+.4f}, "
            f"realized {realized:+.4f} — modelled the wrong direction"
        )

    diff = realized - expected
    if abs(diff) <= tolerance_usd:
        return True, ""
    if expected != 0.0 and abs(diff) <= abs(expected) * rel_tolerance:
        return True, ""

    return False, (
        f"{obs.venue} funding MAGNITUDE mismatch: expected {expected:+.4f}, "
        f"realized {realized:+.4f} (diff {diff:+.4f}) — check interval "
        f"(8h vs hourly) / rate vs price-denominated"
    )
