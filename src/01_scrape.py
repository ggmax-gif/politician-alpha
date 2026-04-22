"""
Stage 1 — Acquire raw inputs.

Two sources, both fetched via the unit-5 toolkit (`requests` + parsing):

1. **Senate trades** — downloaded as a single bulk JSON from the
   `timothycarambat/senate-stock-watcher-data` GitHub mirror. The mirror
   stitches together every Periodic Transaction Report filed under the
   STOCK Act and exposes one tidy record per disclosed trade.

2. **S&P 500 constituents** — scraped from the Wikipedia "List of S&P 500
   companies" page using `requests` + BeautifulSoup + regex. Provides the
   ticker -> GICS sector mapping we use as the benchmark sector weighting
   later in the pipeline. This is the part of the pipeline that exercises
   the unit-5 HTML scraping workflow on a live, real-world page.

Outputs:
    data/raw/senate_all_transactions.json   # cached raw download
    data/raw/trades_raw.csv                 # flattened, one row per trade
    data/raw/sp500_constituents.csv         # ticker, name, sector, sub_industry
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SENATE_URL = (
    "https://raw.githubusercontent.com/timothycarambat/"
    "senate-stock-watcher-data/master/aggregate/all_transactions.json"
)
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

HEADERS = {"User-Agent": "BEE2041-student-research/1.0 (academic; University of Exeter)"}


# --------------------------------------------------------------------------- #
# Senate trades (JSON mirror)
# --------------------------------------------------------------------------- #

def fetch_senate_trades() -> pd.DataFrame:
    """Download the Senate trades JSON (cached) and return it as a DataFrame."""
    out_json = RAW / "senate_all_transactions.json"
    if out_json.exists():
        records = json.loads(out_json.read_text())
    else:
        print(f"[senate] GET {SENATE_URL}")
        r = requests.get(SENATE_URL, headers=HEADERS, timeout=120)
        r.raise_for_status()
        records = r.json()
        out_json.write_text(json.dumps(records))
    df = pd.DataFrame.from_records(records)
    print(f"[senate] {len(df):,} raw rows, {df['senator'].nunique()} senators")
    return df


# --------------------------------------------------------------------------- #
# S&P 500 constituents (Wikipedia HTML scrape)
# --------------------------------------------------------------------------- #

# Wikipedia leaves footnote anchors like "[1]" or "[a]" inside cell text — strip
# them so the symbol/sector strings round-trip cleanly downstream.
FOOTNOTE_RE = re.compile(r"\[[^\]]+\]")

# yfinance uses '-' where Wikipedia (and CRSP) use '.', e.g. BRK.B -> BRK-B.
TICKER_DOT_RE = re.compile(r"\.")


def _clean(text: str) -> str:
    return FOOTNOTE_RE.sub("", text).strip()


def scrape_sp500_constituents() -> pd.DataFrame:
    """Scrape the Wikipedia constituents table and return a clean DataFrame."""
    print(f"[sp500] GET {SP500_URL}")
    r = requests.get(SP500_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("constituents table not found — Wikipedia layout changed")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [_clean(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if len(cells) < 4:
            continue
        symbol_raw, name, sector, sub_industry = cells[0], cells[1], cells[2], cells[3]
        rows.append(
            {
                "symbol": TICKER_DOT_RE.sub("-", symbol_raw),
                "symbol_raw": symbol_raw,
                "name": name,
                "gics_sector": sector,
                "gics_sub_industry": sub_industry,
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    print(f"[sp500] {len(df):,} constituents across {df['gics_sector'].nunique()} sectors")
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    trades = fetch_senate_trades()
    trades.to_csv(RAW / "trades_raw.csv", index=False)
    print(f"[senate] -> {RAW / 'trades_raw.csv'}")

    sp500 = scrape_sp500_constituents()
    sp500.to_csv(RAW / "sp500_constituents.csv", index=False)
    print(f"[sp500] -> {RAW / 'sp500_constituents.csv'}")


if __name__ == "__main__":
    main()
