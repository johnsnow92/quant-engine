"""Cross-venue funding hunt — where does carry actually pay, beyond Kraken?

Kraken's own funding was too thin (~3-5%/yr gross) to beat fees. This scans
perpetual funding across every venue Coinalyze covers (30+), ranks the richest
PERSISTENT funding (mean over a lookback window + sign-persistence, not a noisy
snapshot), annualizes for fair cross-venue comparison, and flags which venues
are actually executable for us.

Annualization: per-interval funding x settlements/year. Hourly-funding venues
(Kraken, Hyperliquid, dYdX, Vertex, Aevo, Paradex) settle 8760x/yr; the rest
default to 8h (1095x/yr). This is an estimate — verify a candidate's native
interval before trusting its annualized figure.

A rich + persistent + same-sign funding stream on a venue we can trade is the
carry the Kraken-only test couldn't find. Capturing it still needs spot (or
inverse-perp) hedge execution ON that venue and must clear that venue's fees.

Usage:
    uv run python scripts/run_venue_funding_scan.py
    uv run python scripts/run_venue_funding_scan.py --top 20 --min-persistence 0.8
"""
from __future__ import annotations

import argparse
import time

from quant_engine.data.coinalyze import CoinalyzeClient, CoinalyzeError

# Venues we can plausibly execute on (have/can-get an API): Kraken now,
# Coinbase via existing CDP creds, Hyperliquid (open API).
EXECUTABLE = {"Kraken", "Coinbase", "Coinbase International", "Hyperliquid"}
# EXECUTABLE means "we can reach the API" — NOT permission to trade. Live orders
# are gated by quant_engine.execution.venue_legality (default-deny, Michigan
# operator): Coinalyze's "Coinbase" feed is the International (INTX) perp
# exchange, which that gate explicitly BLOCKS, and Hyperliquid is off-allowlist.
# Only CFTC-regulated Kraken futures clears the gate today.
US_LEGAL = {"Kraken"}
HOURLY_VENUES = {"Kraken", "Hyperliquid", "Coinbase", "Coinbase International",
                 "dYdX", "Vertex", "Aevo", "Paradex"}
# Coinalyze mis-ingests Kraken funding (inconsistent absolute/relative units per
# instrument), so Kraken is scanned via its NATIVE feed (data/kraken.py), never
# through Coinalyze. Coinalyze 'value' is the per-interval rate in PERCENT.
COINALYZE_UNRELIABLE = {"Kraken"}
LOOKBACK_DAYS = 30


def settlements_per_year(venue: str) -> int:
    return 8760 if venue in HOURLY_VENUES else 1095  # hourly vs 8h default


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-venue funding hunt")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--min-persistence", type=float, default=0.75,
                        help="min fraction of days funding held its sign")
    parser.add_argument("--quotes", nargs="+", default=["USDT", "USD"])
    parser.add_argument("--all-venues", action="store_true",
                        help="scan every venue (slow, rate-limited); default = executable venues only")
    args = parser.parse_args()

    try:
        cv = CoinalyzeClient()
        exch = cv.fetch_exchanges()
        markets = cv.fetch_future_markets()
    except CoinalyzeError as exc:
        print(f"Coinalyze unavailable: {exc}")
        return

    perps = markets[
        markets["is_perpetual"]
        & markets["has_ohlcv_data"]
        & markets["quote_asset"].isin(args.quotes)
    ].copy()
    perps["venue"] = perps["exchange"].map(exch).fillna(perps["exchange"])
    perps = perps[~perps["venue"].isin(COINALYZE_UNRELIABLE)]  # Kraken via native feed only
    if not args.all_venues:
        perps = perps[perps["venue"].isin(EXECUTABLE)]
    symbols = perps["symbol"].tolist()
    scope = "ALL venues" if args.all_venues else f"executable venues ({', '.join(sorted(EXECUTABLE))})"
    print(f"Scanning {len(symbols)} perpetual markets — {scope} (quotes: {', '.join(args.quotes)})...")
    if not symbols:
        print("No markets matched. Try --all-venues or different --quotes.")
        return

    # --- current funding snapshot, paced well under the rate limit ---------
    snap = []
    for i in range(0, len(symbols), 20):  # Coinalyze caps symbols/request (~20)
        snap.append(cv.fetch_current_funding(symbols[i:i + 20]))
        time.sleep(2.0)  # < 40 req/min
    import pandas as pd
    funding_now = pd.concat(snap, ignore_index=True)
    funding_now = funding_now.merge(
        perps[["symbol", "base_asset", "venue"]], on="symbol", how="left")

    # candidate extremes by |current funding|
    funding_now["abs_fr"] = funding_now["funding_rate"].abs()
    candidates = funding_now.sort_values("abs_fr", ascending=False).head(args.top * 4)

    # --- persistence over the lookback via funding history -----------------
    now = int(time.time())
    frm = now - LOOKBACK_DAYS * 86_400
    cand_syms = candidates["symbol"].tolist()
    hist_frames = []
    for i in range(0, len(cand_syms), 20):
        hist_frames.append(cv.fetch_funding_history(cand_syms[i:i + 20], "daily", frm, now))
        time.sleep(2.0)
    hist = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()

    rows = []
    for sym, grp in hist.groupby("symbol"):
        fr = grp["funding_rate"]
        mean_fr = float(fr.mean())
        if mean_fr == 0:
            continue
        persistence = float((fr.apply(lambda x: (x > 0) == (mean_fr > 0))).mean())
        meta = candidates[candidates["symbol"] == sym].iloc[0]
        venue = meta["venue"]
        # Coinalyze 'value' is a PERCENT per interval -> /100 to a fraction.
        ann = (mean_fr / 100.0) * settlements_per_year(venue)
        rows.append({
            "symbol": sym, "base": meta["base_asset"], "venue": venue,
            "mean_fr_per_int": mean_fr, "ann_funding": ann, "persistence": persistence,
            "executable": venue in EXECUTABLE, "us_legal": venue in US_LEGAL,
        })

    res = pd.DataFrame(rows)
    res = res[res["persistence"] >= args.min_persistence]
    res = res.reindex(res["ann_funding"].abs().sort_values(ascending=False).index).head(args.top)

    print(f"\nRichest PERSISTENT funding (|ann| desc, persistence >= {args.min_persistence:.0%}, "
          f"{LOOKBACK_DAYS}d lookback):")
    hdr = f"{'base':<7}{'venue':<16}{'ann_funding':>13}{'persist':>9}{'side':>16}{'exec':>6}{'legal':>7}"
    print(hdr); print("-" * len(hdr))
    for _, r in res.iterrows():
        side = "short perp" if r["ann_funding"] > 0 else "long perp"
        print(f"{r['base']:<7}{r['venue']:<16}{r['ann_funding']:>12.1%}{r['persistence']:>9.0%}"
              f"{side:>16}{'YES' if r['executable'] else '--':>6}"
              f"{'YES' if r['us_legal'] else '--':>7}")

    ex = res[res["us_legal"]]
    print(f"\nLegally executable opportunities (venue_legality gate): {len(ex)}")
    print("Annualized funding is gross, pre-fee, and assumes you HOLD the carry; "
          "net it against that venue's spot+perp round-trip fees and basis risk "
          "(re-run the carry backtest with the candidate's real funding series).")


if __name__ == "__main__":
    main()
