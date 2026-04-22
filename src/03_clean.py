"""
Clean the scraped trades and join party affiliation.

Inputs:
    data/raw/trades_raw.csv
    data/interim/prices.csv, sectors.csv, rf.csv
    congress-legislators YAML (downloaded on first run)

Outputs:
    data/processed/trades_clean.csv
    data/processed/legislators.csv
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raise NotImplementedError("Fill in once 02_fetch_prices.py is producing outputs.")


if __name__ == "__main__":
    main()
