# LEAN CLI workspace (QuantConnect)

Wires the [QuantConnect LEAN](https://www.quantconnect.com/docs/v2/lean-cli) engine
into quant-engine. LEAN is the documented upgrade path from the bootstrap
backtester (`src/quant_engine/backtest/engine.py`) and — unlike Breakout/Kraken
Prop — can execute live on real crypto brokerages (Kraken, Coinbase, Binance,
Bybit), which makes it the candidate execution layer for the funding-carry work.

## Status
- [x] CLI installed: `lean 1.0.225` (via `uv tool install lean`, on PATH at `~/.local/bin`).
- [x] Workspace dir + gitignore wired into quant-engine.
- [ ] **BLOCKED — login needs a paid org.** Verified 2026-06-13: the API token
      (required by `lean login` AND `lean init`) is paywalled — the account page
      returns "To request an access token, you must belong to a paid
      organization." A free QuantConnect account cannot use the LEAN CLI at all.

### Paths to unblock (pick one)
1. **Upgrade QuantConnect** to a paid tier (~$20/mo) → get the token → full CLI
   (cloud + local backtests, live on Kraken/Coinbase). Cleanest.
2. **Free: run open-source LEAN via Docker directly**, bypassing the CLI/token —
   `docker run quantconnect/lean` with a local `config.json` + algorithm. No
   account needed; more manual (LEAN-format algos + data).
3. **Park it.** The Python carry engine already covers the research; revisit LEAN
   when committing to a strategy + capital.

## One-time setup

1. **Log in** (do this yourself — it's a credential):
   ```
   ! lean login
   ```
   Get your **user-id** and **API token** from
   https://www.quantconnect.com/account (Security → request/copy API token).
   Credentials are stored in `~/.lean/credentials` (outside this repo).

2. **Scaffold the workspace** (after login, run from this `lean/` dir):
   ```
   cd ~/Dev/quant-engine/lean && lean init --language python
   ```

3. **(Optional) Local engine** — only for `lean backtest`/`lean live` locally:
   ```
   brew install --cask docker   # then launch Docker Desktop once
   ```
   Skip this to run everything in the cloud (`lean cloud ...`).

## Common commands
| Goal | Command |
|---|---|
| New algorithm | `lean project-create "carry-research" --language python` |
| Cloud backtest (no Docker) | `lean cloud push --project "carry-research" && lean cloud backtest "carry-research" --open` |
| Local backtest (needs Docker) | `lean backtest "carry-research"` |
| Pull cloud projects locally | `lean cloud pull` |
| Live (real brokerage) | `lean live deploy "carry-research" --brokerage "Kraken" ...` |

## How it connects to the carry engine
The validated research lives in `scripts/run_*_carry.py` + `strategies/carry_signal.py`.
LEAN is the path to (a) event-driven backtests with real fills/slippage and (b)
live execution on a supported brokerage — the cross-venue execution layer the
pure-Python research couldn't provide. Port a strategy by translating the carry
signal into a LEAN `QCAlgorithm` once a candidate clears the research gate.
