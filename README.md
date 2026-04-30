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

> Did stocks bought by US Senators between 2012 and 2020 outperform the market,
> and is the alpha concentrated in particular parties, sectors, or seniority cohorts?

## Data sources

| Source | What we get | Access |
|---|---|---|
| [`timothycarambat/senate-stock-watcher-data`](https://github.com/timothycarambat/senate-stock-watcher-data) | Aggregated JSON of every Senate Periodic Transaction Report (2012–2020) | Downloaded in `src/01_scrape.py` |
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

# 3. Render the interactive HTML blog (Plotly + ToC + tabset)
make html                       # produces notebooks/blog.html

# 4. (Optional) explore the executed notebook in Jupyter
jupyter notebook notebooks/blog.ipynb
```

The four pipeline stages are also wrapped as a single `make all` target;
see `make help` for the full list.

All steps are idempotent, rerunning skips files that already exist.

## Project layout

```
.
├── README.md
├── PROJECT_SPEC.md          # Build spec used to scaffold the project
├── Makefile                 # `make all`, `make html`, `make help`, ...
├── requirements.txt
├── src/
│   ├── 01_scrape.py         # Senate trades JSON + Wikipedia S&P 500 scrape
│   ├── 02_fetch_prices.py   # yfinance: prices, S&P 500 benchmark, risk-free rate
│   ├── 03_clean.py          # Join party from legislators YAML, compute forward returns
│   └── 04_analysis.py       # CAPM regression + causal forest + figures
├── notebooks/
│   ├── blog.qmd             # Quarto source (proofread; render with `make html`)
│   └── blog.ipynb           # Executed notebook (kept in sync via `quarto convert`)
└── data/
    ├── raw/                 # Senate JSON, Wikipedia scrape
    ├── interim/             # yfinance caches
    └── processed/           # Cleaned, labelled analytic dataset + regression tables
```

## Environment requirements

Python 3.11+ and the packages pinned in [`requirements.txt`](requirements.txt)
(pandas, numpy, requests, beautifulsoup4, lxml, pyyaml, yfinance, statsmodels,
scikit-learn, econml, matplotlib, seaborn, plotly, jupyter, nbformat, tqdm).
Rendering `blog.html` additionally requires [Quarto](https://quarto.org)
(`brew install quarto`).

## Caveat

Correlation is not causation. This project measures *realised* performance of
disclosed trades, not the counterfactual of what senators would have earned
without their information set. Alpha estimates are noisy and sensitive to the
estimation window, sector weights, and how we treat multi-owner (`Spouse`,
`Joint`, `Dependent Child`) filings. The blog post discusses these caveats in full.
