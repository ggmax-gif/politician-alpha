# BEE2041 Empirical Project — Build Spec for GitHub Copilot

You are GitHub Copilot working inside a graded undergraduate dissertation repo. Build the project end-to-end, one stage at a time. **Stop and wait for confirmation between stages.** Make small, frequent git commits — never bundle multiple stages into one commit.

---

## 1. Context

- **Course:** BEE2041 Data Science in Economics (University of Exeter, 2026)
- **Weight:** 70% of final grade
- **Deadline:** 2026-05-01, 15:00 (UK time)
- **Toolkit constraint:** the course taught `requests` + `BeautifulSoup` + regex for scraping, `pandas` for cleaning, `statsmodels` for OLS/CAPM, and `econml` for causal forests. **Stay inside this toolkit unless I approve a deviation.** No Selenium, Playwright, undetected-chromedriver, or third-party data APIs.
- **Working directory** is a git repo. Commit early, commit often.

---

## 2. Research question

> Did stocks bought by US Senators between 2014 and 2024 outperform the market, and is any alpha concentrated in particular parties, sectors, or seniority cohorts?

Two-track analysis:

1. **OLS / CAPM** (`statsmodels`) — headline Jensen's alpha by party.
2. **Causal forest** (`econml.dml.CausalForestDML` or `econml.grf.CausalForest`) — heterogeneous treatment effects (CATE) within each party. Treatment = "trade was a senator purchase"; control = a synthetic same-sector S&P 500 trade on the same date. The forest is the centrepiece of the unit-5 modelling component, so give it real care.

---

## 3. Data sources (use exactly these — do not substitute)

| # | Source | Access |
|---|---|---|
| 1 | **Senate trades — `https://www.capitoltrades.com/trades`** | Live HTML scrape with `requests` + `BeautifulSoup` + regex. Filter to chamber = Senate after parsing (URL filters are JS-only). Paginate via `?page=N` (12 rows/page, ~3,000 pages, expect 30–50 min on first cold run; **must cache to disk so re-runs are instant**). This is a real scrape on a real site — it is what satisfies the unit-5 web-scraping requirement. |
| 2 | **S&P 500 constituents — `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`** | Scrape the `<table id="constituents">` for ticker → GICS sector mapping. Strip footnote anchors with regex; normalise tickers for yfinance (`BRK.B` → `BRK-B`). |
| 3 | **Party affiliation — `unitedstates/congress-legislators`** | Fetch `legislators-current.yaml` and `legislators-historical.yaml` raw from GitHub via `requests`, parse with `pyyaml`. |
| 4 | **Daily prices, S&P 500 benchmark (`^GSPC`), risk-free rate (`^IRX` 13-week T-bill, annualised)** | `yfinance`. |

**Forbidden substitutions:**
- The `senate-stock-watcher-data` GitHub mirror — using someone else's pre-aggregated JSON would not exercise the unit-5 scraping skill.
- House Clerk PDFs — too fragile for a graded deadline.
- Any paid API (Bloomberg, Refinitiv, Quiver Quant paid tier).

---

## 4. Repo layout (preserve, don't restructure)

```
.
├── README.md
├── PROJECT_SPEC.md            # this file
├── Makefile
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_scrape.py           # CapitolTrades senate trades + Wikipedia S&P 500
│   ├── 02_fetch_prices.py     # yfinance: prices + benchmark + risk-free
│   ├── 03_clean.py            # legislators YAML join, amount parsing, forward returns
│   └── 04_analysis.py         # CAPM regression + causal forest + figures
├── notebooks/
│   └── blog.ipynb             # 1,000–2,500 word blog post with 4–7 figures
└── data/
    ├── raw/                   # scraper outputs (gitignored — regenerable)
    ├── interim/               # yfinance caches (gitignored)
    └── processed/             # analytic dataset, regression tables, figures (committed)
```

`requirements.txt` (minimum):
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

---

## 5. Git workflow — read this carefully

**Commit early, commit often.** This is a graded project and the marker may inspect commit history; a sensible cadence demonstrates engineering discipline.

### Commit cadence rules

- **One commit per logical unit of work**, not one commit per stage. A single stage will typically generate 3–8 commits.
  - Examples of one logical unit:
    - "add the page-walker function for the scrape"
    - "fix off-by-one in the forward-return calculation"
    - "add the CAPM regression table writer"
- **Never** lump scrape + price-fetch + clean into one commit.
- **Never** commit broken code. If a stage's script doesn't run end-to-end, fix it before committing.
- **Always** run the script and eyeball the output (`df.head()`, row counts, file sizes) before committing.

### Commit message format

Conventional Commits: `<type>(<scope>): <subject>` — subject in imperative mood, ≤72 chars.

Types: `feat` (new behaviour), `fix` (bug fix), `refactor` (no behaviour change), `docs`, `chore`, `test`, `data` (one-off data regeneration).

Scope: stage name (`scrape`, `prices`, `clean`, `analysis`, `notebook`) or `repo` for cross-cutting changes.

Examples:
```
feat(scrape): add CapitolTrades page walker with disk cache
fix(scrape): handle senators with multi-state representation history
refactor(clean): extract amount-range parser into helper
docs(readme): document the make targets
```

If a commit needs explanation beyond the subject, add a short body (wrap at 72 chars) explaining **why**, not what.

### Before each commit

1. `git status` — confirm only the files you intended are staged.
2. `git diff --staged` — eyeball the diff one more time.
3. Run the script you changed, end-to-end. Confirm outputs.
4. Commit.

### What to gitignore

- `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`
- `data/raw/*` (large, regenerable) — but keep `.gitkeep`
- `data/interim/` (caches)
- `empiricalProject_2026.pdf` (course brief — not redistributable)
- Keep `data/processed/` committed so the marker can run the notebook offline.

---

## 6. Pipeline stages

Each script is **idempotent at the file level** (skip work whose output already exists). All paths derived from `Path(__file__).resolve().parents[1]`. PEP 8, type hints, docstring at the top of each file explaining inputs/outputs.

Use a polite `User-Agent` header for every outbound request:
```python
HEADERS = {"User-Agent": "BEE2041-student-research/1.0 (academic; University of Exeter)"}
```

### Stage 1 — `src/01_scrape.py`

Two functions, both real network calls.

#### 1a. `scrape_capitoltrades(max_pages: int | None = None) -> pd.DataFrame`

Walk `https://www.capitoltrades.com/trades?page=N` from page 1 onward.

- Extract the single `<table>` per page; columns are: Politician, Traded Issuer, Published, Traded, Filed After, Owner, Type, Size, Price.
- Parse the politician cell into separate fields: `politician_name`, `party` (Democrat/Republican/Independent), `chamber` (Senate/House), `state`. The cell concatenates them as one string — split on visible whitespace and known party tokens.
- Parse the issuer cell into `issuer_name` and `ticker` (the `XYZ:US` suffix).
- Parse `Traded` (transaction date, e.g. `25 Mar 2026`) and `Published` (publication date) as ISO dates.
- Parse `Size` (e.g. `1K–15K`, `15K–50K`) into `amount_low_usd`, `amount_high_usd`.
- **Filter to `chamber == "Senate"`** before saving (the URL filter is JS-only, so we filter Python-side after parsing).
- **Disk cache per page**: write each page's parsed rows to `data/raw/capitoltrades/page_NNNN.csv` immediately after fetching. On re-run, skip pages already cached. This makes the scrape resumable after a network blip.
- Polite delay: `time.sleep(0.5)` between requests. Random jitter (±0.2s) is fine.
- Stop when a page returns zero rows OR when `max_pages` is reached (the parameter exists so the user can smoke-test with `max_pages=5` before committing to the full ~3,000-page run).
- After the walk completes, concatenate every cached page CSV into `data/raw/trades_raw.csv`.

#### 1b. `scrape_sp500_constituents() -> pd.DataFrame`

GET the Wikipedia page, `BeautifulSoup(..., "lxml")`, find `<table id="constituents">`. For each row:

- Extract symbol, security name, GICS sector, GICS sub-industry.
- Strip footnote anchors (`[1]`, `[a]`) using regex.
- Normalise tickers for yfinance (`BRK.B` → `BRK-B`).

Save to `data/raw/sp500_constituents.csv` with columns: `symbol, symbol_raw, name, gics_sector, gics_sub_industry`.

#### Stage 1 commit cadence

Suggested:
1. `feat(scrape): add Wikipedia S&P 500 constituent scraper` (the easy one first)
2. `feat(scrape): add CapitolTrades page parser` (parses one page in isolation)
3. `feat(scrape): add resumable page walker with disk cache`
4. `fix(scrape): split politician cell into name/party/chamber/state`
5. `feat(scrape): concatenate cached pages into trades_raw.csv`

### Stage 2 — `src/02_fetch_prices.py`

- Build the universe: union of (a) tickers in `trades_raw.csv` and (b) S&P 500 symbols.
- Use `yfinance.download(...)` in batches of ~50 tickers, `auto_adjust=True`, period `2012-01-01` → `2025-06-30` (covers trade window + 12-month tail for forward returns).
- Cache long-format `date, ticker, adj_close` to `data/interim/prices.csv`.
- Fetch and cache:
  - `^GSPC` daily → `data/interim/benchmark.csv`
  - `^IRX` daily (annualised %) → `data/interim/rf.csv`
- Failed tickers (delisted, mergers like UTX → RTX) → `data/interim/missing_tickers.txt`. Don't crash; log and continue.

### Stage 3 — `src/03_clean.py`

1. Download both legislators YAML files from `unitedstates/congress-legislators`. Build `{normalised_name → party}` for senators, handling multi-term party-switch cases (use the term covering the trade date).
2. Filter `trades_raw.csv` to legitimate stock trades (drop options, bonds, PDF-only filings if any).
3. Join party (CapitolTrades already gives us a party label — cross-check it against the YAML and log mismatches).
4. Compute forward returns at horizons **30, 90, 180 trading days** using cached prices: `ret_h = adj_close[t+h] / adj_close[t] - 1`. Same for the benchmark → `bench_ret_h`. Excess return = `ret_h - bench_ret_h`.
5. Join GICS sector from `sp500_constituents.csv` (left join — non-S&P trades get `sector = "Other"`).
6. Output `data/processed/trades_labelled.csv` — one row per trade, all features needed for analysis.

### Stage 4 — `src/04_analysis.py`

**A. CAPM by party.** For each (party × horizon):
- Regress `r_i - r_f = α + β·(r_m - r_f) + ε` with HC3 robust standard errors.
- Save `data/processed/capm_results.csv` with columns: `party, horizon, alpha, alpha_se, beta, beta_se, t_alpha, p_alpha, n`.

**B. Causal forest.** Purchases only:
- Outcome `Y` = 90-day excess return.
- Treatment `T` = 1 (senator purchase) vs 0 (synthetic same-sector S&P stock on the same date). Build the matched control panel inside the script.
- Features `X`: sector one-hot, party one-hot, log(amount_mid), senator-tenure-years, day-of-week, market-volatility (20-day rolling σ of `^GSPC`).
- Fit `CausalForestDML` (or `CausalForest`); report ATE and per-party CATE; save per-trade CATE estimates to `data/processed/cate_per_trade.csv` and feature importance to `data/processed/cate_feature_importance.csv`.

**Figures** (matplotlib/seaborn, save to `data/processed/figures/`): cumulative excess returns by party, alpha bar chart with CIs, CATE distribution by party, sector decomposition. 4–7 figures total — these get reused in the notebook.

### Stage 5 — `Makefile`

Targets: `setup`, `scrape`, `prices`, `clean`, `analyse`, `all`, `notebook`, `clean-data` (deletes `data/interim/` and `data/processed/` but **not** `data/raw/`). `all` chains the four pipeline scripts.

### Stage 6 — `notebooks/blog.ipynb`

A 1,000–2,500 word blog embedding 4–7 of the stage-4 figures. Structure:

1. Hook — concrete claim with one striking number.
2. Why this matters — STOCK Act, public-trust angle.
3. Data + method — short, plain-English; link to the repo for replication.
4. Headline result — CAPM alpha by party.
5. The real story — causal forest reveals heterogeneity (which sectors / senators drive it).
6. Caveats — selection effects, disclosure lag, amount-range censoring, no counterfactual identification.
7. Reproducibility footer — `make all`.

Voice: undergraduate analyst, not hedge-fund marketer. Hedge appropriately ("consistent with", "suggests", not "proves").

---

## 7. Engineering rules

- **No mocks.** Hit the real URLs. The scrape and price-fetch must work against live sources.
- **No silent failures.** If a download fails, log loudly and continue; don't write empty files that pretend success.
- **Comment policy:** comment the *why*, not the *what*. Don't narrate code that reads itself. Multi-paragraph docstrings only on module headers.
- **Verification gates:** at the end of each stage, print row counts, show `df.head()`, confirm joined tables don't lose >50% of rows silently.
- **Update `README.md`** at the end of each stage to reflect any drift from this spec.

---

## 8. Acceptance criteria

- `make all` runs from a clean checkout (after `pip install -r requirements.txt`) and produces every file under `data/processed/`.
- `notebooks/blog.ipynb` runs top-to-bottom without errors and renders all figures inline.
- The CAPM table has α, β, t-stat, p-value, n for each (party × horizon) cell.
- The causal-forest section reports ATE + per-party CATE and at least one figure showing heterogeneity.
- README's "Reproducing the analysis" section matches the actual commands.
- Git log shows a sensible commit cadence — small, scoped commits with conventional-commit messages.

---

**Start with stage 1, sub-step 1a (Wikipedia S&P 500 scraper — the simpler one first). Show me the diff and the script's output, then ask before continuing.**
