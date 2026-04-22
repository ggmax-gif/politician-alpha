PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: all setup scrape prices clean analyse notebook clean-data help

## Default target
all: scrape prices clean analyse

## Create venv and install dependencies
setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	@echo "Environment ready. Run: make all"

## Stage 1 — Senate JSON download + Wikipedia S&P 500 scrape
scrape:
	$(PYTHON) src/01_scrape.py

## Stage 2 — Fetch prices, benchmark and risk-free rate via yfinance
prices:
	$(PYTHON) src/02_fetch_prices.py

## Stage 3 — Join party, parse amounts, compute forward returns
clean:
	$(PYTHON) src/03_clean.py

## Stage 4 — CAPM regression + causal forest + figures
analyse:
	$(PYTHON) src/04_analysis.py

## Launch the blog notebook
notebook:
	.venv/bin/jupyter notebook notebooks/blog.ipynb

## Delete derived outputs but keep raw data caches (avoids re-downloading)
clean-data:
	rm -f data/processed/*.csv
	rm -rf data/processed/figures/
	rm -rf data/interim/
	@echo "Cleared data/processed/ and data/interim/. Raw downloads intact."

help:
	@echo ""
	@echo "Targets:"
	@echo "  make setup       Create venv and install requirements.txt"
	@echo "  make all         Run full pipeline (scrape → prices → clean → analyse)"
	@echo "  make scrape      Stage 1: Senate trades + Wikipedia scrape"
	@echo "  make prices      Stage 2: yfinance prices, benchmark, risk-free"
	@echo "  make clean       Stage 3: join party, forward returns, sector labels"
	@echo "  make analyse     Stage 4: CAPM + causal forest + figures"
	@echo "  make notebook    Open blog.ipynb in Jupyter"
	@echo "  make clean-data  Delete processed outputs (keep raw caches)"
	@echo ""
