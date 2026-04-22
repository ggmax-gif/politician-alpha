"""
Stage 2 — Fetch historical prices, benchmark, and risk-free rate via yfinance.

Inputs:
    data/raw/trades_raw.csv             # universe contributor: senator-traded stocks
    data/raw/sp500_constituents.csv     # universe contributor: S&P 500 names

Outputs:
    data/interim/prices.csv             # long-format: date, ticker, adj_close
    data/interim/benchmark.csv          # ^GSPC daily adj_close (the market)
    data/interim/rf.csv                 # ^IRX daily annualised yield (risk-free)
    data/interim/missing_tickers.txt    # tickers yfinance returned nothing for

Idempotent at the file level: each output is regenerated only if missing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# Trade dates run 2012-09 -> 2020-11. We need ~12 months of tail beyond the
# last trade so 30/90/180-day forward returns are computable on every row,
# and a small lead so any future estimation window has burn-in.
START = "2012-01-01"
END = "2022-06-30"

BATCH_SIZE = 50  # tickers per yfinance.download call — polite + tolerant of one-off failures


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #

def build_universe() -> list[str]:
    """Union of senator-traded stock tickers and S&P 500 constituents."""
    trades = pd.read_csv(RAW / "trades_raw.csv")
    senate_tickers = trades.loc[trades.asset_type == "Stock", "ticker"].dropna()
    senate_tickers = {t.strip() for t in senate_tickers if isinstance(t, str)}
    senate_tickers -= {"", "--", "N/A"}

    sp500 = pd.read_csv(RAW / "sp500_constituents.csv")
    sp500_tickers = set(sp500.symbol.dropna())

    universe = sorted(senate_tickers | sp500_tickers)
    print(f"[universe] {len(universe):,} tickers "
          f"(senate-traded: {len(senate_tickers)}, sp500: {len(sp500_tickers)})")
    return universe


# --------------------------------------------------------------------------- #
# Price download
# --------------------------------------------------------------------------- #

def _download_batch(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Download a batch; return (long-format prices, list of empty tickers)."""
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        actions=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"]), list(tickers)

    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    # When a single ticker is requested, yfinance returns a flat (non multi-index)
    # frame. Normalise that case so the loop below is uniform.
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tickers[0]: raw}, axis=1)

    for tkr in tickers:
        if tkr not in raw.columns.get_level_values(0):
            missing.append(tkr)
            continue
        sub = raw[tkr][["Close"]].dropna()
        if sub.empty:
            missing.append(tkr)
            continue
        rows.append(
            sub.reset_index().rename(columns={"Date": "date", "Close": "adj_close"})
               .assign(ticker=tkr)[["date", "ticker", "adj_close"]]
        )

    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["date", "ticker", "adj_close"]
    )
    return long, missing


def fetch_prices(universe: list[str]) -> None:
    out = INTERIM / "prices.csv"
    if out.exists():
        print(f"[prices] {out} exists — skip")
        return

    INTERIM.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    missing: list[str] = []

    for i in tqdm(range(0, len(universe), BATCH_SIZE), desc="batches"):
        batch = universe[i:i + BATCH_SIZE]
        long, miss = _download_batch(batch)
        parts.append(long)
        missing.extend(miss)

    prices = pd.concat(parts, ignore_index=True)
    prices.to_csv(out, index=False)
    print(f"[prices] {len(prices):,} rows, {prices.ticker.nunique()} tickers -> {out}")

    if missing:
        miss_path = INTERIM / "missing_tickers.txt"
        miss_path.write_text("\n".join(sorted(set(missing))) + "\n")
        print(f"[prices] {len(set(missing))} tickers had no data -> {miss_path}")


# --------------------------------------------------------------------------- #
# Benchmark and risk-free
# --------------------------------------------------------------------------- #

def fetch_single(symbol: str, out_name: str, value_col: str) -> None:
    """Download a single yfinance series and save it as a 2-col CSV."""
    out = INTERIM / out_name
    if out.exists():
        print(f"[{symbol}] {out} exists — skip")
        return

    df = yf.download(
        symbol, start=START, end=END,
        auto_adjust=True, progress=False, actions=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    series = df["Close"].dropna()
    if isinstance(series, pd.DataFrame):  # yfinance sometimes wraps in 1-col DF
        series = series.iloc[:, 0]
    out_df = series.reset_index()
    out_df.columns = ["date", value_col]
    out_df.to_csv(out, index=False)
    print(f"[{symbol}] {len(out_df):,} rows -> {out}")


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    universe = build_universe()
    fetch_prices(universe)
    fetch_single("^GSPC", "benchmark.csv", "sp500_close")
    fetch_single("^IRX", "rf.csv", "tbill_yield_pct")


if __name__ == "__main__":
    main()
