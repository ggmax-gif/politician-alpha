"""
Portfolio construction, sector-weight comparison, and CAPM regression.

Inputs:
    data/processed/trades_clean.csv

Outputs:
    data/processed/sector_weights.csv
    data/processed/capm_results.csv
    data/processed/figures/*.png
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raise NotImplementedError("Fill in after 03_clean.py produces processed trades.")


if __name__ == "__main__":
    main()
