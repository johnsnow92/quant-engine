"""Historical backtest of the spot-perpetual funding-rate carry strategy.

Strategy mechanics:
  - Hold (1-α)·D in spot, α·D as perpetual margin (short perp, delta-neutral).
  - Spot price changes cancel against the short perp — net price exposure = 0.
  - Hourly P&L: funding_rate × (1-α) × equity  (carry on the spot leg).
  - Liquidation: if p_t/p_entry ≥ r_liq(α) the short perp is force-closed;
    equity loses LGD fraction and the position re-enters at the current price.
  - Rebalance: every rebal_hours, α* is re-computed from rolling realized vol
    and rolling mean funding rate via risk-constrained bisection.
  - Warm-up: the first vol_lookback_days × 24 hours accumulate data but hold
    no position (equity stays flat), eliminating any look-ahead bias in σ_h.

Reference: arXiv:2605.05089 (risk-constrained collateral control).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from quant_engine.analysis.collateral_sizing import (
    alpha_risk_constrained,
    liq_barrier,
    theta_F_from_max_leverage,
)


@dataclass
class BacktestConfig:
    """Parameters governing one backtest run."""
    eps: float = 0.05           # max acceptable liquidation prob per review
    stress_mult: float = 1.0    # vol stress multiplier applied to rolling σ_h
    lgd: float = 1.0            # fraction of net margin (α−θ_F)·equity lost on liquidation
                                # 1.0 = full margin wipeout (conservative); typical HL: ~0.8-1.0
    rebal_hours: int = 24       # rebalance α* every N hours
    vol_lookback_days: int = 30 # rolling window for σ_h (days)
    carry_lookback_days: int = 30  # rolling window for funding carry estimate (days)
    h_hours: float = 24.0       # review horizon passed to risk-constrained optimizer
    theta_F: float = 0.01       # venue maintenance margin fraction (HL 50×: 1/100)
    recenter_on_rebalance: bool = True  # reset the perp entry reference to the current
                                # price at each rebalance (model topping up / withdrawing
                                # margin). Makes the 24h-horizon ε control valid by
                                # preventing multi-period drift accumulation. Set False
                                # to model a fixed-entry position that rides to liquidation.

    # --- Cost model ---
    fee_bps_per_side: float = 4.5  # taker fee + slippage per leg per trade, in bps.
                                # Hyperliquid base taker ≈ 0.045% (4.5bp); maker ≈ 1.5bp.
                                # Charged on both legs (spot + perp) at open, rebalance
                                # turnover, liquidation roundtrip, and final close.
    funding_capture: float = 1.0  # execution-friction fraction of POSITIVE funding
                                # actually received. On Hyperliquid funding is paid
                                # mechanically with no spread → 1.0. The real funding
                                # drag (negative-funding hours, empirically ~15% of gross)
                                # is intrinsic to the signed series, not this knob. Lower
                                # only to model receipt slippage on a different venue.
    spot_fee_bps_per_side: float | None = None  # spot-leg fee/side; None → fee_bps_per_side.
                                # Venues with asymmetric legs differ sharply: Kraken Pro
                                # spot taker ≈ 40bp (maker 25bp) vs the perp leg below.
    perp_fee_bps_per_side: float | None = None  # perp-leg fee/side; None → fee_bps_per_side.
                                # Bitnomial US perps ≈ $0.15/contract ≈ 2bp at current
                                # BTC/ETH prices — an order of magnitude under spot.

    @property
    def spot_fee_bps(self) -> float:
        """Resolved spot-leg fee (falls back to the symmetric fee_bps_per_side)."""
        return self.fee_bps_per_side if self.spot_fee_bps_per_side is None else self.spot_fee_bps_per_side

    @property
    def perp_fee_bps(self) -> float:
        """Resolved perp-leg fee (falls back to the symmetric fee_bps_per_side)."""
        return self.fee_bps_per_side if self.perp_fee_bps_per_side is None else self.perp_fee_bps_per_side

    @property
    def vol_lookback_hours(self) -> int:
        return self.vol_lookback_days * 24

    @property
    def carry_lookback_hours(self) -> int:
        return self.carry_lookback_days * 24


@dataclass
class BacktestResult:
    """Output of one backtest run."""
    asset: str
    config: BacktestConfig
    n_hours: int
    n_active_hours: int          # hours actually in position (post warm-up)

    # Equity curve (length = n_hours; flat during warm-up at 1.0)
    equity_series: np.ndarray
    alpha_series: np.ndarray     # α* at each hour
    timestamps_ms: np.ndarray    # ms timestamps

    # Liquidation events
    liq_indices: list[int]

    # Summary metrics (active period only)
    total_return: float          # equity[-1] - 1.0
    ann_return: float            # CAGR over active period
    ann_vol: float               # annualised std of hourly log-returns
    sharpe: float                # ann_return / ann_vol
    max_drawdown: float          # max peak-to-trough equity drop

    # Carry decomposition (as fraction of initial equity)
    gross_carry: float           # positive-funding income received (the gross opportunity)
    liq_losses: float            # total equity lost to liquidations
    net_pnl: float               # total_return ≈ gross_carry − funding_drag − fees − liq_losses
    fees: float = 0.0            # total trading fees paid (open + turnover + exit + liq roundtrips)
    funding_drag: float = 0.0    # negative-funding paid + receipt-execution friction

    @property
    def liq_count(self) -> int:
        return len(self.liq_indices)

    def summary(self) -> str:
        lines = [
            f"Asset={self.asset}  ε={self.config.eps:.0%}  stress={self.config.stress_mult:.1f}×  "
            f"lgd={self.config.lgd:.0%}",
            f"  Active: {self.n_active_hours}h ({self.n_active_hours/24:.0f}d)",
            f"  Return: {self.total_return:.2%}  Ann: {self.ann_return:.2%}  "
            f"Vol: {self.ann_vol:.2%}  Sharpe: {self.sharpe:.2f}",
            f"  MaxDD: {self.max_drawdown:.2%}  Liq events: {self.liq_count}",
            f"  Gross carry: {self.gross_carry*10000:.1f}bps  "
            f"Funding drag: {self.funding_drag*10000:.1f}bps  "
            f"Fees: {self.fees*10000:.1f}bps  "
            f"Liq losses: {self.liq_losses*10000:.1f}bps  "
            f"Net P&L: {self.net_pnl*10000:.1f}bps",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def merge_price_funding(
    price_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align hourly candles and funding on the same hour-bucket timestamps.

    Both DataFrames need columns: ts (ms).
    price_df additionally needs: close.
    funding_df additionally needs: funding_rate.
    """
    p = price_df[["ts", "close"]].copy()
    f = funding_df[["ts", "funding_rate"]].copy()
    p["hour_ms"] = (p["ts"] // 3_600_000) * 3_600_000
    f["hour_ms"] = (f["ts"] // 3_600_000) * 3_600_000
    p = p.drop_duplicates("hour_ms").sort_values("hour_ms")
    f = f.drop_duplicates("hour_ms").sort_values("hour_ms")
    merged = pd.merge(p[["hour_ms", "close"]], f[["hour_ms", "funding_rate"]],
                      on="hour_ms", how="inner").sort_values("hour_ms").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_backtest(
    price_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    asset: str,
    config: BacktestConfig | None = None,
    event_log: list | None = None,
) -> BacktestResult:
    """Run the carry strategy backtest on aligned price + funding data.

    Args:
        price_df: DataFrame with columns [ts(ms), close].
        funding_df: DataFrame with columns [ts(ms), funding_rate].
        asset: ticker label.
        config: BacktestConfig (uses defaults if None).
        event_log: optional list; if provided, one dict per liquidation event
            is appended with full path context (entry/trigger price, the move
            that breached the barrier, α and r_liq held, the σ_h that sized α).

    Returns:
        BacktestResult with equity curve, alpha series, and summary metrics.
    """
    if config is None:
        config = BacktestConfig()

    merged = merge_price_funding(price_df, funding_df)
    n = len(merged)
    if n < config.vol_lookback_hours + 2:
        raise ValueError(
            f"Insufficient data: {n} hours, need at least "
            f"{config.vol_lookback_hours + 2} for warm-up + simulation"
        )

    prices = merged["close"].to_numpy(dtype=float)
    fundings = merged["funding_rate"].to_numpy(dtype=float)
    timestamps = merged["hour_ms"].to_numpy(dtype=float)

    equity_series = np.ones(n)
    alpha_series = np.zeros(n)
    liq_indices: list[int] = []

    warm_up = config.vol_lookback_hours
    equity = 1.0
    entry_price = prices[warm_up]
    entry_index = warm_up

    # Conservative initial α: just above the minimum viable level
    alpha_crit = config.theta_F / (1.0 + config.theta_F)
    current_alpha = alpha_crit + 0.05

    gross_carry = 0.0
    liq_losses = 0.0
    last_sigma_stressed = float("nan")  # σ_h·stress that sized the current α

    # Cost-model accumulators. Each rebalance/open/close trades BOTH legs, so the
    # per-turnover cost is the sum of the two leg fees (spot on Kraken, perp on
    # Bitnomial). When both legs share one fee this equals the old fee_rate×2.
    leg_fee_rate = (config.spot_fee_bps + config.perp_fee_bps) / 10_000.0
    total_fees = 0.0
    funding_drag = 0.0
    prev_notional_frac = 0.0  # per-leg notional as a fraction of equity (0 = flat)

    for i in range(n):
        if i < warm_up:
            equity_series[i] = equity
            alpha_series[i] = 0.0
            continue

        price = prices[i]
        funding = fundings[i]

        # --- Rebalance α* ---
        if (i - warm_up) % config.rebal_hours == 0:
            price_win = prices[max(0, i - config.vol_lookback_hours): i + 1]
            if len(price_win) > 1:
                log_rets = np.log(price_win[1:] / price_win[:-1])
                sigma_h = float(log_rets.std(ddof=1))
                fund_win = fundings[max(0, i - config.carry_lookback_hours): i]
                kappa_h = max(float(fund_win.mean()), 0.0) if len(fund_win) > 0 else 0.0
                new_alpha = alpha_risk_constrained(
                    config.eps,
                    sigma_h * config.stress_mult,
                    config.h_hours,
                    config.theta_F,
                )
                if not math.isnan(new_alpha):
                    current_alpha = new_alpha
                    last_sigma_stressed = sigma_h * config.stress_mult
                _ = kappa_h  # available for future use (e.g., dynamic ε scaling)
            # Re-center the perp margin: reset the entry reference to the current
            # price so r_spot only accumulates over one review window, matching the
            # 24h horizon the barrier was sized for.
            if config.recenter_on_rebalance:
                entry_price = price
                entry_index = i

        # --- Trading fees: turnover from opening / rebalancing the position ---
        # Per-leg notional = (1−α)·equity on both spot and perp. Only a CHANGE in
        # the target fraction is a trade (the position rides equity drift for free);
        # at the first active bar prev=0 so this charges opening both legs.
        target_notional_frac = 1.0 - current_alpha
        turnover_frac = abs(target_notional_frac - prev_notional_frac)
        if turnover_frac > 0.0:
            fee = leg_fee_rate * turnover_frac * equity  # spot + perp legs
            equity -= fee
            total_fees += fee
        prev_notional_frac = target_notional_frac

        # --- Liquidation check ---
        # Only a genuine leveraged short (r_liq > 1, i.e. α > α_crit) can be
        # liquidated. When α ≤ α_crit the barrier sits at/below entry and there
        # is no real perp position to blow up — skip the check entirely.
        r_spot = price / entry_price
        r_liq_val = liq_barrier(current_alpha, config.theta_F)
        if r_liq_val > 1.0 and r_spot >= r_liq_val:
            # Physical loss: margin deployed minus maintenance margin returned.
            # Net margin = (α - θ_F)·equity; lgd scales how much of that is lost
            # (default 1.0 = full margin wipeout, conservative).
            net_margin_fraction = max(0.0, current_alpha - config.theta_F)
            loss = config.lgd * net_margin_fraction * equity
            if event_log is not None:
                event_log.append({
                    "index": i,
                    "timestamp_ms": float(timestamps[i]),
                    "entry_index": entry_index,
                    "entry_timestamp_ms": float(timestamps[entry_index]),
                    "hours_held": i - entry_index,
                    "entry_price": float(entry_price),
                    "trigger_price": float(price),
                    "pct_move": float(r_spot - 1.0),
                    "alpha": float(current_alpha),
                    "r_liq": float(r_liq_val),
                    "barrier_move_pct": float(r_liq_val - 1.0),
                    "sigma_h_stressed": float(last_sigma_stressed),
                    "equity_before": float(equity),
                    "loss": float(loss),
                })
            equity -= loss
            liq_losses += loss
            entry_price = price
            entry_index = i
            liq_indices.append(i)
            # Re-size immediately after liquidation
            price_win = prices[max(0, i - config.vol_lookback_hours): i + 1]
            if len(price_win) > 1:
                sigma_h = float(np.log(price_win[1:] / price_win[:-1]).std(ddof=1))
                new_alpha = alpha_risk_constrained(
                    config.eps,
                    sigma_h * config.stress_mult,
                    config.h_hours,
                    config.theta_F,
                )
                if not math.isnan(new_alpha):
                    current_alpha = new_alpha
            # Liquidation roundtrip cost: force-close the blown perp and reopen
            # the position (≈ two legs of trading on the post-loss equity).
            liq_fee = leg_fee_rate * (1.0 - current_alpha) * equity
            equity -= liq_fee
            total_fees += liq_fee
            prev_notional_frac = 1.0 - current_alpha  # avoid re-charging next bar

        # --- Funding: decompose into received (positive hours) vs paid (negative) ---
        # The short receives funding when it is positive and PAYS when negative.
        # gross_carry tracks the positive-funding opportunity; funding_drag tracks
        # the negative-funding paid PLUS any execution friction on receipts.
        # On Hyperliquid funding is paid mechanically (no spread), so funding_capture
        # defaults to 1.0 — the negative-funding drag is intrinsic to the signed series.
        signed_funding = funding * (1.0 - current_alpha) * equity
        received = max(signed_funding, 0.0)
        paid = max(-signed_funding, 0.0)
        capture_loss = received * (1.0 - config.funding_capture)  # friction on receipts
        equity += signed_funding - capture_loss
        gross_carry += received
        funding_drag += paid + capture_loss

        # --- Final close: exit both legs at the last active bar ---
        if i == n - 1:
            exit_fee = leg_fee_rate * (1.0 - current_alpha) * equity
            equity -= exit_fee
            total_fees += exit_fee

        equity_series[i] = equity
        alpha_series[i] = current_alpha

    # --- Metrics over active period ---
    active = equity_series[warm_up:]
    n_active = len(active)

    log_rets_active = np.diff(np.log(np.maximum(active, 1e-30)))
    ann_vol = float(log_rets_active.std(ddof=1)) * math.sqrt(8760.0) if len(log_rets_active) > 1 else 0.0
    total_return = float(active[-1]) - 1.0
    ann_return = float(active[-1]) ** (8760.0 / n_active) - 1.0 if n_active > 0 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    running_max = np.maximum.accumulate(active)
    max_drawdown = float(np.max(1.0 - active / np.maximum(running_max, 1e-30)))

    return BacktestResult(
        asset=asset,
        config=config,
        n_hours=n,
        n_active_hours=n_active,
        equity_series=equity_series,
        alpha_series=alpha_series,
        timestamps_ms=timestamps,
        liq_indices=liq_indices,
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        gross_carry=gross_carry,
        liq_losses=liq_losses,
        net_pnl=total_return,
        fees=total_fees,
        funding_drag=funding_drag,
    )


# ---------------------------------------------------------------------------
# Multi-config batch
# ---------------------------------------------------------------------------

def run_backtest_grid(
    price_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    asset: str,
    eps_levels: tuple[float, ...] = (0.01, 0.05, 0.10),
    stress_mults: tuple[float, ...] = (1.0, 1.5, 2.0),
    base_config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    """Run backtests for all (ε, stress) combinations."""
    base = base_config or BacktestConfig()
    results = []
    for eps in eps_levels:
        for mult in stress_mults:
            cfg = replace(base, eps=eps, stress_mult=mult)
            try:
                results.append(run_backtest(price_df, funding_df, asset, cfg))
            except ValueError as exc:
                print(f"  [warn] {asset} ε={eps:.0%} stress={mult:.1f}×: {exc}")
    return results


def results_to_table(results: list[BacktestResult]) -> str:
    """Format a grid of backtest results as a summary table."""
    header = (
        f"{'Asset':>6} {'ε':>5} {'Stress':>6}  "
        f"{'AnnRet':>8} {'Sharpe':>7} {'Liq':>4}  "
        f"{'Gross':>8} {'FundDrag':>9} {'Fees':>7} {'LiqLoss':>8} {'NetPnL':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.asset:>6} {r.config.eps:>5.0%} {r.config.stress_mult:>5.1f}×  "
            f"{r.ann_return:>8.2%} {r.sharpe:>7.2f} {r.liq_count:>4}  "
            f"{r.gross_carry*1e4:>6.0f}bp {r.funding_drag*1e4:>7.0f}bp "
            f"{r.fees*1e4:>5.0f}bp {r.liq_losses*1e4:>6.0f}bp {r.net_pnl*1e4:>6.0f}bp"
        )
    return "\n".join(rows)
