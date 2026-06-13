# quant-engine

Crypto / derivatives quant engine — bootstrap vertical slice.

A reproducible pipeline from market data to a guarded paper trade:

```
Crypto.com public REST  ->  strategy signal  ->  funding-aware backtest  ->  paper broker (guarded)
```

**Not investment advice. No live capital until backtest and paper-trade both pass.**

## What's here

| Layer | Module | Notes |
|---|---|---|
| Data | `data/cryptocom.py`, `data/coindesk.py` | Perp OHLCV + funding (Crypto.com) and second-venue funding (Binance via CoinDesk), no auth, CSV-cached |
| Strategy | `strategies/ma_crossover.py`, `strategies/funding_tilt.py`, `strategies/carry_signal.py` | Trivial directional signals + the carry direction picker |
| Backtest | `backtest/engine.py`, `backtest/carry.py`, `backtest/metrics.py` | Directional + **market-neutral (perp+spot) carry**, funding-aware, no look-ahead, `price_beta` neutrality check |
| Cross-venue | `analysis/funding_spread.py` | Normalize venues to per-hour, funding spread, cross-venue carry |
| Execution | `execution/guards.py`, `execution/paper_broker.py` | Pre-trade limits + simulated fills |

## Setup & run

```bash
cd ~/Dev/quant-engine
uv sync --extra dev                       # create venv, install deps + package (editable)

uv run pytest                             # unit tests (offline, deterministic)
uv run python scripts/run_backtest.py     # live: fetch BTCUSD-PERP 1h, backtest both strategies
uv run python scripts/run_carry.py        # directional tilt vs market-neutral carry (price_beta)
uv run python scripts/cross_venue_funding.py  # Crypto.com vs Binance funding spread + carry
uv run python scripts/paper_demo.py       # paper broker + guard rejections
```

`run_backtest.py` caches fetched data to `data/cache/` so reruns are offline and
reproducible. Use `--no-cache` to force a fresh pull, or `--instrument`,
`--timeframe`, `--count`, `--fee-bps` to vary the run.

## Known simplifications (next iteration targets)

- **`funding_tilt` is a directional tilt** (still price-exposed). The
  market-neutral version is `backtest/carry.py` + `strategies/carry_signal.py`,
  which hedges the price leg with spot (`run_carry.py` shows `price_beta` ≈ 0).
  Still single-venue; shorting spot for the `s = -1` side assumes borrow.
- **Cross-venue carry per-hour accrual is an approximation.** Funding settles at
  discrete times per venue, and cross-venue basis/transfer/borrow frictions are
  not modeled — `cross_venue_funding.py` sizes the spread, it isn't fill-accurate.
- **Funding-rate unit is venue-specific.** Verify Crypto.com's funding
  convention (per-hour vs per-interval) before trusting carry PnL magnitudes.
- **Backtest engine is the bootstrap one.** Upgrade path is `vectorbt` (vectorized)
  or QuantConnect **LEAN** (full multi-asset engine with live trading) once
  strategies stabilize.
- **No slippage / partial-fill model** in the paper broker yet — fills at order
  price. Add a slippage model before trusting fill assumptions.

## Path to live (do not skip)

1. Backtest green on out-of-sample history.
2. Paper-trade green against the live feed for a meaningful window.
3. Live execution adapter (exchange API) wired **behind the same `PreTradeGuard`**.
4. Start with a hard notional cap and the kill switch one toggle away.
