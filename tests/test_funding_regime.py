"""Tests for the funding-regime watcher — focus on the carry trigger.

The watcher had no test coverage and it gates a capital lane, so this also pins
the existing directional (15%/24h) behaviour. All funding data is mocked; no
network. Crypto.com funding is fed as a constant or recent/older split so the
24h tail and the 7d tail can be controlled independently.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_engine.watchers.funding_regime import RegimeState, check_regime

_HOURS_PER_YEAR = 8_760.0


def _const_funding(annual: float, n: int = 200) -> pd.DataFrame:
    """A funding frame whose every 1h bar annualizes to ``annual``."""
    per_hour = annual / _HOURS_PER_YEAR
    return pd.DataFrame({"ts": range(n), "funding_rate": [per_hour] * n})


def _split_funding(recent_annual: float, older_annual: float,
                   recent_bars: int = 24, n: int = 200) -> pd.DataFrame:
    """Recent ``recent_bars`` annualize to recent_annual; the rest to older_annual.

    Lets the 24h tail differ from the 7d tail so the directional and carry windows
    can be exercised in isolation.
    """
    per_recent = recent_annual / _HOURS_PER_YEAR
    per_older = older_annual / _HOURS_PER_YEAR
    rates = [per_older] * (n - recent_bars) + [per_recent] * recent_bars
    return pd.DataFrame({"ts": range(n), "funding_rate": rates})


def _check(cc_btc: pd.DataFrame, cc_eth: pd.DataFrame | None = None, **kwargs) -> RegimeState:
    cc_eth = cc_eth if cc_eth is not None else _const_funding(0.0)
    cc_mock = MagicMock()
    cc_mock.fetch_funding.side_effect = (
        lambda sym, count: cc_btc if "BTC" in sym else cc_eth
    )
    with patch("quant_engine.watchers.funding_regime.CryptoComClient", return_value=cc_mock), \
         patch("quant_engine.watchers.funding_regime.CoinDeskClient",
               side_effect=RuntimeError("COINDESK_API_KEY unset")):
        return check_regime(**kwargs)


def _has_trigger(state: RegimeState, prefix: str) -> bool:
    """Return whether a structured trigger label starts with ``prefix``."""
    return any(trigger.startswith(prefix) for trigger in state.triggered_by)


# ---------------------------------------------------------------------------
# Carry trigger (5% APR / 7d) — the recalibration this change is about
# ---------------------------------------------------------------------------

def test_carry_regime_on_at_six_percent():
    state = _check(_const_funding(0.06))
    assert state.is_on
    assert state.btc_carry_ann == pytest.approx(0.06, abs=1e-6)
    assert _has_trigger(state, "BTC carry ")
    # 6% is below the 15% directional hurdle, so ONLY carry fires.
    assert not _has_trigger(state, "BTC Crypto.com ")


def test_carry_regime_off_below_five_percent():
    state = _check(_const_funding(0.03))
    assert state.is_on is False
    assert state.triggered_by == []


def test_carry_regime_on_at_exact_five_percent_boundary():
    # The threshold is inclusive (>=); 5%/8760*8760 round-trips to exactly 0.05.
    state = _check(_const_funding(0.05))
    assert state.is_on
    assert state.btc_carry_ann == pytest.approx(0.05, abs=1e-6)
    assert _has_trigger(state, "BTC carry ")


def test_carry_uses_7d_window_not_24h():
    # Last 24h spike to 20%, but the 7d mean stays ~2.9% → directional ON, carry OFF.
    state = _check(_split_funding(recent_annual=0.20, older_annual=0.0))
    assert state.is_on
    assert state.btc_cc_ann == pytest.approx(0.20, abs=1e-6)     # 24h tail
    assert state.btc_carry_ann < 0.05                            # 7d tail
    assert _has_trigger(state, "BTC Crypto.com ")
    assert not _has_trigger(state, "BTC carry ")


# ---------------------------------------------------------------------------
# Signed (not abs): negative funding is not a short-carry opportunity
# ---------------------------------------------------------------------------

def test_negative_funding_does_not_trigger_carry():
    state = _check(_const_funding(-0.06))
    assert state.btc_carry_ann == pytest.approx(-0.06, abs=1e-6)
    assert not _has_trigger(state, "BTC carry ")
    assert state.is_on is False   # |-6%| < 15% directional too


def test_large_negative_triggers_directional_but_not_carry():
    state = _check(_const_funding(-0.20))
    assert state.is_on                                      # |-20%| >= 15% directional
    assert _has_trigger(state, "BTC Crypto.com ")
    assert not _has_trigger(state, "BTC carry ")


# ---------------------------------------------------------------------------
# Config + reporting
# ---------------------------------------------------------------------------

def test_carry_threshold_is_configurable():
    assert _check(_const_funding(0.04)).is_on is False                       # default 5%
    assert _check(_const_funding(0.04), carry_threshold=0.03).is_on is True  # lowered


def test_summary_reports_carry_line():
    summary = _check(_const_funding(0.06)).summary()
    assert "BTC carry" in summary
    assert "(7d)" in summary


def test_non_positive_lookback_raises():
    # A negative lookback would flip pandas tail(-n) semantics ("all but last n").
    with pytest.raises(ValueError, match="positive"):
        _check(_const_funding(0.06), lookback=0)
    with pytest.raises(ValueError, match="positive"):
        _check(_const_funding(0.06), carry_lookback=-1)
