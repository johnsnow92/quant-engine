# Plan 02 — Coinbase Advanced Trade Live Executor

**Status:** APPROVED — gate cleared (32 tests pass, paper_demo.py green)  
**Date:** 2026-06-13  
**Author:** Claude Code

---

## Goal

Add a `CoinbaseBroker` that sits behind the same `PreTradeGuard` as `PaperBroker`
and routes orders to Coinbase Advanced Trade (INTX perpetuals) via direct REST —
no MCP, no subprocess, no LLM in the hot path.

The engine's execution layer becomes swappable: pass `broker=PaperBroker(...)` in
tests, pass `broker=CoinbaseBroker(...)` in production, same interface.

---

## Gate Status

Per CLAUDE.md rule: "No live execution adapter until backtest + paper-trade are
both green."

- [x] 32 unit tests pass (`uv run pytest`)
- [x] `scripts/paper_demo.py` shows fills and guard rejections
- [x] Funding regime watcher live in GitHub Actions

Gate is cleared. Implementation may proceed.

---

## Architecture

```
RegimeState.is_on == True
        │
        ▼
  carry_signal() → Order(instrument, side, qty, price)
        │
        ▼
  PreTradeGuard.check()  ← kill-switch, notional cap, allowlist
        │  raises GuardRejection → log + stop
        ▼
  CoinbaseBroker.submit_order()
        │
        ├── POST /api/v3/brokerage/orders  (market order)
        │
        ├── poll GET /api/v3/brokerage/orders/{id}  (up to 3s)
        │
        └── return Fill(instrument, side, qty, fill_price, fee)
```

---

## Interface Contract

`CoinbaseBroker` must satisfy the same duck-type as `PaperBroker`:

```python
def submit_order(self, order: Order) -> Fill: ...
def position(self, instrument: str) -> float: ...
def equity(self, marks: dict[str, float]) -> float: ...
```

No changes to `Order`, `Fill`, `PreTradeGuard`, or `PaperBroker`.

---

## Instrument Mapping

Quant-engine internally uses Crypto.com convention.
Coinbase INTX perpetual product IDs differ:

| Internal name    | Coinbase product_id  |
|------------------|----------------------|
| `BTCUSD-PERP`    | `BTC-PERP-INTX`      |
| `ETHUSD-PERP`    | `ETH-PERP-INTX`      |

The `CoinbaseBroker` owns this map. Orders for unmapped instruments raise
`GuardRejection` (caught before hitting the API).

---

## Authentication

Coinbase Advanced Trade uses **CDP API keys** (JWT / ES256, not OAuth).

Required env vars (load from Infisical at startup — never hardcode):

```
COINBASE_CDP_API_KEY_NAME    e.g. organizations/{org}/apiKeys/{id}
COINBASE_CDP_PRIVATE_KEY     EC private key PEM (base64-encoded in Infisical)
```

Use the `coinbase-advanced-py` SDK (`coinbase-advanced-py>=1.7`) which handles
JWT signing internally. The engine never imports the Coinbase MCP tools — this is
a plain REST call so the engine runs standalone in GitHub Actions or cron.

---

## Order Flow

1. `submit_order(order)` called
2. Guard check: `guard.check(order, current_position)` — raises `GuardRejection` on reject
3. Map internal instrument → Coinbase product_id (raise `GuardRejection` if unmapped)
4. Build Coinbase order:
   - `order_type`: `MARKET`
   - `side`: `BUY` / `SELL`
   - `product_id`: from instrument map
   - `order_configuration.market_market_ioc.base_size`: `str(order.qty)`
   - `client_order_id`: `uuid4()` (idempotency key)
5. POST to `/api/v3/brokerage/orders`
6. Poll `GET /api/v3/brokerage/orders/{order_id}` up to 10× at 0.3 s intervals
   until status in `{FILLED, CANCELLED, FAILED}`
7. On FILLED: extract `average_filled_price`, `total_fees`, compute `Fill`
8. Update `self._positions[instrument]` local cache
9. Return `Fill`

---

## Risk Limits (Tranche 1)

Baked into the `PreTradeGuard` constructor in `run_live.py`.
Change only via code review — never at runtime.

| Parameter             | Tranche 1 value                       |
|-----------------------|---------------------------------------|
| `max_notional_usd`    | 2,000                                 |
| `max_position_qty`    | 0.025 BTC / 0.25 ETH                  |
| `allowed_instruments` | `{BTCUSD-PERP, ETHUSD-PERP}` (internal) |
| `kill_switch`         | `False` (set `True` to halt all orders) |

---

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| `GuardRejection` | Re-raise — caller logs and skips this cycle |
| HTTP 4xx (bad request, insufficient funds) | Raise `ExecutionError(non_retriable=True)` |
| HTTP 429 / 5xx / network timeout | Retry 3× with 1 s backoff, then raise `ExecutionError` |
| Order polled CANCELLED or FAILED | Raise `ExecutionError` with order status |
| Position cache stale on startup | Fetch from `GET /api/v3/brokerage/portfolios` on first call |

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `src/quant_engine/execution/coinbase_broker.py` | **Create** — `CoinbaseBroker`, `ExecutionError` |
| `src/quant_engine/execution/__init__.py` | **Modify** — export `CoinbaseBroker` |
| `scripts/run_live.py` | **Create** — entry point: loads regime state, builds live broker, runs one carry cycle |
| `tests/execution/test_coinbase_broker.py` | **Create** — 6 unit tests with mocked HTTP |
| `pyproject.toml` | **Modify** — add `coinbase-advanced-py>=1.7` to dependencies |

---

## `run_live.py` Sketch

```python
regime = check_regime()
if not regime.is_on:
    sys.exit(0)  # regime watcher handles Telegram alert

guard = PreTradeGuard(
    allowed_instruments={'BTCUSD-PERP', 'ETHUSD-PERP'},
    max_notional_usd=2_000.0,
    max_position_qty=0.025,
)
broker = CoinbaseBroker(guard=guard)

signal = carry_signal(regime)   # returns Order or None
if signal:
    fill = broker.submit_order(signal)
    log.info("Filled: %s", fill)
```

`--dry-run` flag skips the `POST` and returns a synthetic `Fill` — used in CI to
verify the full path (guard + broker construction + carry signal) without live capital.

---

## Tests Required (before merge)

- [ ] `test_submit_order_success` — mocked POST + poll returns FILLED
- [ ] `test_guard_rejection_not_sent` — `GuardRejection` stops before any HTTP call
- [ ] `test_instrument_not_in_map` — unmapped instrument raises `GuardRejection`
- [ ] `test_retries_on_500` — 3 retries then raises `ExecutionError`
- [ ] `test_kill_switch_halts` — `kill_switch=True` raises before any I/O
- [ ] `test_position_cache_update` — position updated after successful fill

**Done =** all 6 tests pass + `uv run python scripts/run_live.py --dry-run` exits 0.
