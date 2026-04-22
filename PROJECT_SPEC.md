# BEE2041 Empirical Project — Build Spec

You are implementing a complete data-science pipeline for a final-year undergraduate empirical economics project. Build it end-to-end, one stage at a time, asking for confirmation between stages.

## 1. Context

- **Course:** BEE2041 Data Science in Economics (University of Exeter, 2026)
- **Weight:** 70% of final grade
- **Deadline:** 2026-05-01, 15:00 (UK time)
- **Toolkit constraint:** the course taught `requests` + `BeautifulSoup` + regex for scraping, `pandas` for cleaning, `statsmodels` for OLS/CAPM, and `econml` for causal forests. Stay inside this toolkit unless I approve a deviation.
- **Working directory** is a git repo. Commit after each stage with a short conventional-commit message.

## 2. Research question

> Did stocks bought by US Senators between 2014 and 2021 outperform the market, and is any alpha concentrated in particular parties, sectors, or seniority cohorts?

Two-track analysis:
- **OLS / CAPM** (`statsmodels`) — headline Jensen's alpha by party.
- **Causal forest** (`econml.dml.CausalForestDML` or `econml.grf.CausalForest`) — heterogeneous treatment effects (CATE) within each party. Treatment = "trade was a senator purchase"; control = matched market trades. The forest is the centrepiece of the unit-5 modelling component, so give it real care.

## 3. Data sources (use exactly these — don't substitute)

| # | Source | Access |
|---|---|---|
| 1 | Senate trades aggregate JSON | `https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json` (default branch is `master`, not `main`) |
| 2 | S&P 500 constituents (for GICS sector mapping + benchmark universe) | Scrape `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies` with `requests` + `BeautifulSoup` + regex. **This scrape is non-negotiable** — it satisfies the unit-5 HTML-scraping requirement |
| 3 | Party affiliation by senator | `unitedstates/congress-legislators` YAML files (`legislators-current.yaml` + `legislators-historical.yaml`) — fetch via `requests`, parse with `pyyaml` |
| 4 | Daily prices, S&P 500 benchmark (`^GSPC`), risk-free rate (`^IRX` 13-week T-bill, annualised) | `yfinance` |

Do **not** scrape House Clerk PDFs — fragile and out of scope.

## 4. Repo layout (already partially scaffolded; preserve it)

```
.
├── README.md
├── Makefile
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_scrape.py
│   ├── 02_fetch_prices.py
│   ├── 03_clean.py
│   └── 04_analysis.py
├── notebooks/
│   └── blog.ipynb
└── data/
    ├── raw/        # scraper outputs
    ├── interim/    # yfinance caches
    └── processed/  # analytic dataset + regression tables + figures
```

`requirements.txt` should contain at minimum:
```
pandas>=2.2
numpy>=1.26
requests>=2.32
beautifulsoup4>=4.12
lxml>=5.2
pyyaml>=6.0
yfinance>=0.2.50
statsmodels>=0.14
scikit-learn>=1.5
econml>=0.15
matplotlib>=3.9
seaborn>=0.13
jupyter>=1.1
tqdm>=4.66
```

## 5. Pipeline stages

Each script must be **idempotent** (re-running skips work already cached on disk) and runnable standalone. All paths derived from `Path(__file__).resolve().parents[1]`. PEP 8, type hints, docstring at the top of each file explaining inputs/outputs.

### Stage 1 — `src/01_scrape.py`

Two functions, both real:

1. **`fetch_senate_trades()`**: GET the Senate JSON, cache the raw bytes to `data/raw/senate_all_transactions.json`, then flatten to `data/raw/trades_raw.csv`. Schema (keep all of these as-is, don't rename yet):
   `transaction_date, owner, ticker, asset_description, asset_type, type, amount, comment, senator, ptr_link`

2. **`scrape_sp500_constituents()`**: GET the Wikipedia page, `BeautifulSoup(..., "lxml")`, find `<table id="constituents">`, iterate `<tr>` rows. For each row extract symbol / security name / GICS sector / GICS sub-industry. Use a regex to:
   - strip footnote anchors like `[1]`, `[a]` from cell text;
   - normalise tickers for yfinance: `BRK.B` → `BRK-B` (replace `.` with `-`).
   Save to `data/raw/sp500_constituents.csv` with columns: `symbol, symbol_raw, name, gics_sector, gics_sub_industry`.

Use a polite `User-Agent` header (`"BEE2041-student-research/1.0 (academic; University of Exeter)"`).

### Stage 2 — `src/02_fetch_prices.py`

- Load the union of (a) tickers seen in `trades_raw.csv` filtered to `asset_type == "Stock"` and (b) S&P 500 symbols from `sp500_constituents.csv`.
- Use `yfinance.download(...)` in batches of ~50 tickers, `auto_adjust=True`, period covering the trade window plus 12 months tail (so we can compute forward returns through 2022).
- Cache long-format `date, ticker, adj_close` to `data/interim/prices.csv`.
- Fetch and cache:
  - `^GSPC` daily → `data/interim/benchmark.csv`
  - `^IRX` daily (annualised %) → `data/interim/rf.csv`
- If a ticker fails to download, log it to `data/interim/missing_tickers.txt` and continue — don't crash the run.

### Stage 3 — `src/03_clean.py`

1. Download both legislators YAML files from `unitedstates/congress-legislators` (raw GitHub URLs). Build a lookup `{normalised_name → party}` for senators. Handle the multi-term case (a senator may switch parties; use the term covering the trade date).
2. Filter `trades_raw.csv` to `asset_type == "Stock"` and `type ∈ {"Purchase", "Sale (Full)", "Sale (Partial)"}`.
3. Parse the dollar `amount` range into `amount_low`, `amount_high`, `amount_mid` (geometric mean is fine).
4. Join party. Drop rows where the senator can't be matched (log the count).
5. Compute forward returns at horizons **30, 90, 180 trading days** using cached prices: `ret_h = adj_close[t+h] / adj_close[t] - 1`. Same for the benchmark on the same dates → `bench_ret_h`. Excess return = `ret_h - bench_ret_h`.
6. Join GICS sector from `sp500_constituents.csv` (left join — non-S&P trades get `sector = "Other"`).
7. Output `data/processed/trades_labelled.csv` with one row per trade and all features needed for analysis.

### Stage 4 — `src/04_analysis.py`

Two analyses, both writing tables to `data/processed/` and figures to `data/processed/figures/`.

**A. CAPM by party.** For each party (R, D) and each horizon (30/90/180):
- Regress trade excess returns on benchmark excess return: `r_i - r_f = α + β·(r_m - r_f) + ε`
- Use HC3 robust standard errors.
- Save coefficient table as `capm_results.csv`.

**B. Causal forest.** For purchases only:
- Outcome `Y` = 90-day excess return.
- Treatment `T` = 1 (senator-purchase) vs 0 (synthetic control: same-sector S&P stock on the same date, equally weighted). Build the control panel inside the script.
- Features `X`: sector one-hot, party one-hot, log(amount_mid), senator-tenure-years, day-of-week, market-volatility (20-day rolling σ of `^GSPC`).
- Fit `CausalForestDML` (or `CausalForest`); report ATE and per-party CATE; save SHAP-style feature importance and per-trade CATE estimates as `cate_per_trade.csv`.

Figures (matplotlib/seaborn): cumulative excess returns by party, alpha bar chart with CIs, CATE distribution by party, sector decomposition. 4–7 figures total — they will be reused in the notebook.

### Stage 5 — `Makefile`

Targets: `setup`, `scrape`, `prices`, `clean`, `analyse`, `all`, `notebook`, `clean-data` (deletes derived files but **not** `data/raw/` caches). Each target runs the corresponding `src/0X_*.py`. `all` chains them.

### Stage 6 — `notebooks/blog.ipynb`

A 1,000–2,500 word blog post embedding 4–7 of the figures from stage 4. Structure:
1. Hook — concrete claim with one striking number.
2. Why this matters — STOCK Act, public-trust angle.
3. Data + method — short, plain-English; link to the repo for replication.
4. Headline result — CAPM alpha by party.
5. The real story — causal forest reveals heterogeneity (which sectors / senators drive it).
6. Caveats — selection effects, disclosure lag, amount-range censoring, no counterfactual identification.
7. Reproducibility footer — `make all`.

Write in the voice of an undergraduate analyst, not a hedge-fund marketer. Hedge appropriately ("consistent with", "suggests", not "proves").

## 6. Workflow rules

- **One stage per commit.** Conventional-commit subject (`feat:`, `fix:`, `docs:`, `chore:`).
- **Verify each stage** before moving on: print row counts, show `df.head()`, sanity-check that joined tables don't lose 50% of rows silently.
- **No mocks.** Hit the real URLs. The scrape and the JSON download must work against live sources.
- **No silent failures.** If a download fails, log loudly and continue; don't write empty files that pretend success.
- **Comment policy:** comment the *why*, not the *what*. Don't narrate code that reads itself.
- **Update `README.md`** at the end to reflect any drift from this spec.

## 7. Acceptance criteria

- `make all` runs from a clean checkout (after `pip install -r requirements.txt`) and produces every file under `data/processed/`.
- `notebooks/blog.ipynb` runs top-to-bottom without errors and renders all figures inline.
- The CAPM table has α, β, t-stat, p-value, n for each (party × horizon) cell.
- The causal-forest section reports ATE + per-party CATE and at least one figure showing heterogeneity.
- README's "Reproducing the analysis" section matches the actual commands.

Start with stage 1. Show me the diff, then ask before continuing to stage 2.
