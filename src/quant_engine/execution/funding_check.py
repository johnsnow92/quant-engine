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
    obs: FundingObservation,
    tolerance_usd: float = 1.0,
    rel_tolerance: float = 0.10,
    rate_floor_annual: float = 0.005,
) -> tuple[bool, str]:
    """True iff realized funding matches the stated convention.

    Catches direction (sign) errors and magnitude errors (how an interval or
    denomination mistake shows up). The discriminator is the stated RATE, not the
    dollar amount: when the rate is materially non-zero the check is RELATIVE, so
    a 3x interval error is caught even at sub-dollar micro size where every amount
    is below any absolute tolerance. The absolute tolerance applies only to the
    near-zero-rate (dead-band) leg, where expected funding is ~0 by design and a
    relative band is undefined — there a few cents of realized noise is fine.
    """
    expected = expected_funding_usd(obs)
    realized = obs.realized_funding_usd

    # Near-zero-rate leg (Kalshi dead band): expected ≈ 0 by design; tolerate
    # small realized noise. Keyed on the RATE only — a material rate that happens
    # to have zero expected (zero position/elapsed) must NOT take this path.
    if abs(obs.stated_rate_annual) <= rate_floor_annual:
        if abs(realized) <= tolerance_usd:
            return True, ""
        return False, (
            f"{obs.venue} funding MAGNITUDE mismatch: near-zero rate so expected ≈0 "
            f"({expected:+.4f}), realized {realized:+.4f} — unexpected funding on a ~0-rate leg"
        )

    # Material rate but zero expected (zero position or elapsed) — nothing to
    # reconcile, so the convention can't be verified from this sample.
    if expected == 0.0:
        return False, (
            f"{obs.venue} material funding rate but expected funding is 0 "
            f"(zero position or elapsed) — cannot verify convention from this sample"
        )

    # Material rate: sign, then a size-independent relative band.
    if realized * expected < 0.0:
        return False, (
            f"{obs.venue} funding SIGN mismatch: expected {expected:+.4f}, "
            f"realized {realized:+.4f} — modelled the wrong direction"
        )

    diff = realized - expected
    if abs(diff) <= abs(expected) * rel_tolerance:
        return True, ""

    return False, (
        f"{obs.venue} funding MAGNITUDE mismatch: expected {expected:+.4f}, "
        f"realized {realized:+.4f} (diff {diff:+.4f}, {abs(diff) / abs(expected):.0%} off) "
        f"— check interval (8h vs hourly) / rate vs price-denominated"
    )
