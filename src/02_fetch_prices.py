"""
Fetch historical daily prices and GICS sector metadata via yfinance.

Inputs:
    data/raw/trades_raw.csv (from 01_scrape.py) — provides the ticker universe
    Optional SPY sector weights — scraped from SPDR for the benchmark

Outputs:
    data/interim/prices.csv            # long-format: date, ticker, adj_close
    data/interim/sectors.csv           # ticker -> GICS sector from yfinance
    data/interim/spy_sector_weights.csv  # S&P 500 sector benchmark
    data/interim/rf.csv                # ^IRX 13-week T-bill (annualised yield)

Skips the network where cached CSVs already exist.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"


def main() -> None:
    # TODO: implement after 01_scrape has produced trades_raw.csv
    raise NotImplementedError("Run 01_scrape.py first, then we'll fill this in.")


if __name__ == "__main__":
    main()
