# quant-engine — CLAUDE.md

Crypto / derivatives quant engine. **Python** (python3.12 + uv), not TS — quant
ecosystem (pandas/numpy/vectorbt/LEAN) is Python-native. Idiomatic PEP8
(4-space, snake_case) applies here over the global JS/TS formatting rules.

## Architecture

`Crypto.com public REST -> strategy signal -> funding-aware backtest -> guarded paper broker`

- `src/quant_engine/data/cryptocom.py` — `CryptoComClient`: perp OHLCV + funding, no auth, CSV cache.
- `src/quant_engine/strategies/` — `Strategy` protocol returns positions in [-1, 1]; engine owns PnL/fees/look-ahead.
- `src/quant_engine/backtest/engine.py` — vectorized, funding-aware (`held * (price_return - funding)`), positions shifted to avoid look-ahead.
- `src/quant_engine/backtest/carry.py` — market-neutral perp+spot carry: `s * (funding + h*spot_return - perp_return)`; `metrics.price_beta` confirms ≈0.
- `src/quant_engine/data/coindesk.py` + `analysis/funding_spread.py` — second-venue (Binance) funding via CoinDesk REST; normalize per-hour, build spread, cross-venue carry.
- `src/quant_engine/execution/guards.py` — `PreTradeGuard` is the kill-switch boundary; **every order passes it first**.
- `src/quant_engine/execution/paper_broker.py` — simulated fills; live adapters must sit behind the same guard.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_backtest.py
uv run python scripts/paper_demo.py
```

## Rules

- No live execution adapter until backtest + paper-trade are both green.
- New strategies implement `generate_positions(market) -> pd.Series`; never bake fees/funding into the signal.
- MCP market-data servers (CoinDesk, Crypto.com) are for interactive exploration by the agent — the **engine uses direct REST** so it runs standalone.
- Next-up: verify each venue's funding interval/convention (Crypto.com interval is ASSUMED 1h), discrete funding accrual + cross-venue basis in the cross-venue carry, add slippage model, swap bootstrap backtest for vectorbt/LEAN, then live exchange adapter behind the guard. (Done: market-neutral perp+spot carry; second-venue funding via CoinDesk.)
