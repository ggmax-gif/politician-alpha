"""
Stage 3 — Join, clean, and enrich the raw senate trades.

Inputs:
    data/raw/trades_raw.csv              # from 01_scrape.py
    data/raw/sp500_constituents.csv      # from 01_scrape.py
    data/interim/prices.csv             # from 02_fetch_prices.py
    data/interim/benchmark.csv          # from 02_fetch_prices.py
    data/interim/rf.csv                 # from 02_fetch_prices.py
    legislators YAML (fetched fresh)    # unitedstates/congress-legislators

Steps:
    1. Build senator -> party lookup from legislators YAML (current + historical).
    2. Filter trades to stock purchases/sales with usable tickers.
    3. Parse dollar amount ranges into low / high / mid (geometric mean).
    4. Join party.
    5. Compute 30/90/180-trading-day forward returns and benchmark excess returns.
    6. Join GICS sector.

Output:
    data/processed/trades_labelled.csv  # one row per trade, ready for analysis
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

HEADERS = {"User-Agent": "BEE2041-student-research/1.0 (academic; University of Exeter)"}

LEGISLATORS_URLS = [
    "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/legislators-current.yaml",
    "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/legislators-historical.yaml",
]

HORIZONS = [30, 90, 180]  # trading days

VALID_TYPES = {"Purchase", "Sale (Full)", "Sale (Partial)"}

# --------------------------------------------------------------------------- #
# Name normalisation for fuzzy matching
# --------------------------------------------------------------------------- #

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")


def _norm(name: str) -> str:
    """Lowercase, strip generational suffixes, punctuation and middle initials."""
    n = name.lower()
    n = _SUFFIX_RE.sub(" ", n)
    n = _PUNCT_RE.sub(" ", n)
    tokens = [t for t in n.split() if len(t) > 1]  # drop single-letter initials
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Step 1 — Build senator -> party lookup from legislators YAML
# --------------------------------------------------------------------------- #

def build_party_lookup() -> dict[str, tuple[str, list[dict]]]:
    """Return {normalised_last_name: (party, senate_terms)} for every senator.

    When a senator has terms in multiple parties we keep all terms so the
    caller can pick the party covering the specific trade date.
    """
    all_people: list[dict] = []
    for url in LEGISLATORS_URLS:
        print(f"[legislators] GET {url.split('/')[-1]}")
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        all_people.extend(yaml.safe_load(r.text))

    lookup: dict[str, tuple[str, list[dict]]] = {}
    for person in all_people:
        sen_terms = [t for t in person["terms"] if t["type"] == "sen"]
        if not sen_terms:
            continue
        last = person["name"]["last"]
        key = _norm(last)
        lookup[key] = (sen_terms[-1].get("party", "Unknown"), sen_terms)

    print(f"[legislators] {len(lookup):,} senators indexed")
    return lookup


def party_on_date(sen_terms: list[dict], trade_date: pd.Timestamp) -> str:
    """Return the party covering trade_date; fall back to most recent term."""
    for term in reversed(sen_terms):
        start = pd.Timestamp(term["start"])
        end = pd.Timestamp(term["end"])
        if start <= trade_date <= end:
            return term.get("party", "Unknown")
    return sen_terms[-1].get("party", "Unknown")


def lookup_party(senator: str, trade_date: pd.Timestamp,
                 lookup: dict[str, tuple[str, list[dict]]]) -> str:
    """Match a senator display name to a party via normalised last-name suffix.

    Checks whether the normalised full trade name ends with any known last name,
    preferring the longest match. This handles multi-word last names (Van Hollen)
    and names with middle initials or suffixes (A. Mitchell McConnell Jr.).
    """
    norm_full = _norm(senator)
    best: tuple[str, list[dict]] | None = None
    best_key_len = 0
    for key, val in lookup.items():
        if norm_full.endswith(key) and len(key) > best_key_len:
            best = val
            best_key_len = len(key)
    if best is None:
        return "Unknown"
    return party_on_date(best[1], trade_date)


# --------------------------------------------------------------------------- #
# Step 2 — Filter trades
# --------------------------------------------------------------------------- #

def load_and_filter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["asset_type"] == "Stock"].copy()
    df = df[df["type"].isin(VALID_TYPES)].copy()
    df = df[df["ticker"].notna() & ~df["ticker"].isin(["--", "N/A", ""])].copy()
    print(f"[filter] {before:,} raw rows -> {len(df):,} stock trades with valid tickers")
    return df


# --------------------------------------------------------------------------- #
# Step 3 — Parse amount ranges
# --------------------------------------------------------------------------- #

_AMOUNT_RE = re.compile(r"\$([0-9,]+)\s*-\s*\$([0-9,]+)")


def parse_amount(amount_str: str) -> tuple[float, float, float]:
    m = _AMOUNT_RE.search(str(amount_str))
    if not m:
        return np.nan, np.nan, np.nan
    lo = float(m.group(1).replace(",", ""))
    hi = float(m.group(2).replace(",", ""))
    mid = np.sqrt(lo * hi)  # geometric mean avoids skew from wide ranges
    return lo, hi, mid


# --------------------------------------------------------------------------- #
# Step 5 — Forward returns
# --------------------------------------------------------------------------- #

def build_price_matrix(prices_path: Path) -> pd.DataFrame:
    """Pivot long prices to wide (date x ticker), sorted chronologically."""
    df = pd.read_csv(prices_path, parse_dates=["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="adj_close")
    return wide.sort_index()


def compute_forward_return(
    price_wide: pd.DataFrame,
    ticker: str,
    trade_date: pd.Timestamp,
    h: int,
) -> float:
    """Return the h-trading-day forward return for ticker from trade_date."""
    if ticker not in price_wide.columns:
        return np.nan
    col = price_wide[ticker].dropna()
    if col.empty:
        return np.nan
    pos = col.index.searchsorted(trade_date)
    if pos >= len(col) or pos + h >= len(col):
        return np.nan
    p0 = col.iloc[pos]
    ph = col.iloc[pos + h]
    if p0 == 0 or np.isnan(p0) or np.isnan(ph):
        return np.nan
    return ph / p0 - 1


def annualised_rf_for_horizon(rf_series: pd.Series, trade_date: pd.Timestamp,
                               h: int) -> float:
    """Accumulated risk-free return over h trading days from trade_date.

    ^IRX is the annualised T-bill yield in percent (e.g. 5.2 means 5.2%).
    Convert to a daily rate, compound over h days.
    """
    pos = rf_series.index.searchsorted(trade_date)
    if pos >= len(rf_series):
        return np.nan
    annual_yield_pct = rf_series.iloc[pos]
    if np.isnan(annual_yield_pct):
        return np.nan
    daily_rf = (1 + annual_yield_pct / 100) ** (1 / 252) - 1
    return (1 + daily_rf) ** h - 1


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    out = PROCESSED / "trades_labelled.csv"
    if out.exists():
        print(f"[clean] {out} exists — delete to regenerate")
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Party lookup
    party_lookup = build_party_lookup()

    # 2. Load + filter trades
    trades = load_and_filter(RAW / "trades_raw.csv")
    trades["transaction_date"] = pd.to_datetime(
        trades["transaction_date"], format="%m/%d/%Y", errors="coerce"
    )
    trades = trades.dropna(subset=["transaction_date"])

    # 3. Parse amount
    parsed = trades["amount"].apply(parse_amount)
    trades["amount_low_usd"] = [x[0] for x in parsed]
    trades["amount_high_usd"] = [x[1] for x in parsed]
    trades["amount_mid_usd"] = [x[2] for x in parsed]

    # 4. Join party
    trades["party"] = trades.apply(
        lambda r: lookup_party(r["senator"], r["transaction_date"], party_lookup), axis=1
    )
    unmatched = (trades["party"] == "Unknown").sum()
    print(f"[party] matched {len(trades) - unmatched:,} / {len(trades):,} trades "
          f"({unmatched:,} unmatched -> 'Unknown')")

    # 5. Forward returns
    print("[returns] building price matrix…")
    price_wide = build_price_matrix(INTERIM / "prices.csv")

    bench_raw = pd.read_csv(INTERIM / "benchmark.csv", parse_dates=["date"])
    bench_series = bench_raw.set_index("date")["sp500_close"].sort_index()
    bench_wide = bench_series.rename("^GSPC").to_frame()

    rf_raw = pd.read_csv(INTERIM / "rf.csv", parse_dates=["date"])
    rf_series = rf_raw.set_index("date")["tbill_yield_pct"].sort_index()

    for h in HORIZONS:
        print(f"  horizon {h}d…")
        trades[f"ret_{h}"] = [
            compute_forward_return(price_wide, row.ticker, row.transaction_date, h)
            for row in trades.itertuples(index=False)
        ]
        trades[f"bench_ret_{h}"] = [
            compute_forward_return(bench_wide, "^GSPC", row.transaction_date, h)
            for row in trades.itertuples(index=False)
        ]
        trades[f"rf_{h}"] = [
            annualised_rf_for_horizon(rf_series, row.transaction_date, h)
            for row in trades.itertuples(index=False)
        ]
        trades[f"excess_ret_{h}"] = trades[f"ret_{h}"] - trades[f"rf_{h}"]
        trades[f"bench_excess_{h}"] = trades[f"bench_ret_{h}"] - trades[f"rf_{h}"]

    # 6. Join GICS sector
    sp500 = pd.read_csv(RAW / "sp500_constituents.csv")[
        ["symbol", "gics_sector", "gics_sub_industry"]
    ]
    trades = trades.merge(sp500, left_on="ticker", right_on="symbol", how="left")
    trades["gics_sector"] = trades["gics_sector"].fillna("Other")
    trades["gics_sub_industry"] = trades["gics_sub_industry"].fillna("Other")
    trades = trades.drop(columns=["symbol"], errors="ignore")

    trades.to_csv(out, index=False)
    print(f"\n[clean] {len(trades):,} rows -> {out}")
    print(trades[["senator", "party", "ticker", "gics_sector",
                   "ret_90", "bench_ret_90", "excess_ret_90"]].head(6).to_string())


if __name__ == "__main__":
    main()
