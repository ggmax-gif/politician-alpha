# Politician Alpha: Did US Senators' Stock Trades Beat the Market?

Final empirical project for **BEE2041 — Data Science in Economics** (University of Exeter, 2026).

A data-driven investigation of US Senate stock trading. Trades are sourced from
the [Senate eFD](https://efdsearch.senate.gov) (via a maintained GitHub mirror of all
Periodic Transaction Reports), classified by political party, mapped to GICS
sectors, benchmarked against the S&P 500, and run through:

- a CAPM regression to estimate Jensen's alpha by party (`statsmodels` OLS), and
- a causal forest (`econml`) to surface heterogeneous treatment effects within each
  party, i.e. *which* trades, by *which* senators, drive the headline alpha.

**Blog post (live, interactive):** **<https://ggmax-gif.github.io/politician-alpha/>**


## Research question

> Did stocks bought by US Senators between 2014 and 2021 outperform the market,
> and is the alpha concentrated in particular parties, sectors, or seniority cohorts?

## Data sources

| Source | What we get | Access |
|---|---|---|
| [`timothycarambat/senate-stock-watcher-data`](https://github.com/timothycarambat/senate-stock-watcher-data) | Aggregated JSON of every Senate Periodic Transaction Report (2014–2021) | Downloaded in `src/01_scrape.py` |
| [Wikipedia: List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) | GICS sector / sub-industry mapping for the benchmark universe | Scraped (`requests` + BeautifulSoup + regex) in `src/01_scrape.py` |
| [`unitedstates/congress-legislators`](https://github.com/unitedstates/congress-legislators) | `legislators-current.yaml` + `legislators-historical.yaml` for party affiliation | Downloaded in `src/03_clean.py` |
| [Yahoo Finance](https://finance.yahoo.com) (via `yfinance`) | Daily prices, S&P 500 (`^GSPC`) benchmark, GICS sector lookups | `src/02_fetch_prices.py` |
| [FRED `DTB3`](https://fred.stlouisfed.org/series/DTB3) (via `yfinance` ticker `^IRX`) | 13-week T-bill yield as the risk-free rate for CAPM | `src/02_fetch_prices.py` |

## Reproducing the analysis

```bash
# 1. Clone and install
git clone <repo-url>
cd "Data Science"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline end-to-end (first run pulls ~3 MB of JSON + price data)
python src/01_scrape.py         # Senate JSON + Wikipedia scrape -> data/raw/
python src/02_fetch_prices.py   # yfinance prices/benchmark/rf  -> data/interim/
python src/03_clean.py          # join party, compute returns   -> data/processed/
python src/04_analysis.py       # CAPM + causal forest + plots  -> data/processed/

# 3. Open the blog post
jupyter notebook notebooks/blog.ipynb
```

All steps are idempotent, rerunning skips files that already exist.

## Project layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── 01_scrape.py         # Senate trades JSON + Wikipedia S&P 500 scrape
│   ├── 02_fetch_prices.py   # yfinance: prices, S&P 500 benchmark, risk-free rate
│   ├── 03_clean.py          # Join party from legislators YAML, compute forward returns
│   └── 04_analysis.py       # CAPM regression + causal forest + figures
├── notebooks/
│   └── blog.ipynb           # Final 1,000–2,500 word write-up with 4–7 visualisations
└── data/
    ├── raw/                 # Senate JSON, Wikipedia scrape
    ├── interim/             # yfinance caches
    └── processed/           # Cleaned, labelled analytic dataset + regression tables
```

## Environment requirements

1. pandas
2. numpy
3. requests
4. beautifulsoup4
5. lxml
6. pyyaml
7. yfinance
8. statsmodels
9. scikit-learn
10. econml
11. matplotlib
12. seaborn
13. jupyter
14. tqdm

## Caveat

Correlation is not causation. This project measures *realised* performance of
disclosed trades, not the counterfactual of what senators would have earned
without their information set. Alpha estimates are noisy and sensitive to the
estimation window, sector weights, and how we treat multi-owner (`Spouse`,
`Joint`, `Dependent Child`) filings. The blog post discusses these caveats in full.
