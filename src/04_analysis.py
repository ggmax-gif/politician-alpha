"""
Stage 4 — CAPM regression and causal forest analysis.

Inputs:
    data/processed/trades_labelled.csv   # from 03_clean.py
    data/interim/benchmark.csv           # from 02_fetch_prices.py

Two tracks:
  A. CAPM (statsmodels OLS, HC3 SEs) — Jensen's alpha by party × horizon.
  B. Causal forest (EconML CausalForestDML) — heterogeneous treatment effects
     (CATE) for Democratic vs Republican purchases, uncovering which sectors
     and trade characteristics drive the gap.

Outputs:
    data/processed/capm_results.csv
    data/processed/cate_per_trade.csv
    data/processed/cate_feature_importance.csv
    data/processed/figures/fig1_cumulative_returns.png
    data/processed/figures/fig2_capm_alpha.png
    data/processed/figures/fig3_cate_distribution.png
    data/processed/figures/fig4_cate_by_sector.png
    data/processed/figures/fig5_sector_returns.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
FIGURES = PROCESSED / "figures"

HORIZONS = [30, 90, 180]
PARTY_COLOURS = {"Democrat": "#2166ac", "Republican": "#d6604d"}

sns.set_theme(style="whitegrid", font_scale=1.1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(
        PROCESSED / "trades_labelled.csv", parse_dates=["transaction_date"]
    )
    bench = pd.read_csv(INTERIM / "benchmark.csv", parse_dates=["date"])
    return df, bench


def market_volatility(bench: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-trading-day rolling σ of S&P 500 daily log-returns, indexed by date."""
    prices = bench.set_index("date")["sp500_close"].sort_index()
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std().rename("market_vol_20d")


# --------------------------------------------------------------------------- #
# Track A — CAPM
# --------------------------------------------------------------------------- #

def run_capm(df: pd.DataFrame) -> pd.DataFrame:
    """OLS CAPM with HC3 SEs for each (party, horizon). Returns results table."""
    records = []
    for party in ["Republican", "Democrat"]:
        for h in HORIZONS:
            sub = df[
                df["party"].eq(party)
                & df[f"excess_ret_{h}"].notna()
                & df[f"bench_excess_{h}"].notna()
            ]
            Y = sub[f"excess_ret_{h}"].values
            X = sm.add_constant(sub[f"bench_excess_{h}"].values)
            res = sm.OLS(Y, X).fit(cov_type="HC3")
            records.append(
                {
                    "party": party,
                    "horizon_days": h,
                    "alpha": res.params[0],
                    "alpha_se": res.bse[0],
                    "alpha_t": res.tvalues[0],
                    "alpha_p": res.pvalues[0],
                    "beta": res.params[1],
                    "beta_se": res.bse[1],
                    "n": int(res.nobs),
                }
            )
            print(
                f"  CAPM {party[:1]} h={h:3d}d: "
                f"α={res.params[0]:+.4f} (t={res.tvalues[0]:+.2f}, "
                f"p={res.pvalues[0]:.3f}), β={res.params[1]:.3f}, n={int(res.nobs)}"
            )

    capm_df = pd.DataFrame(records)
    capm_df.to_csv(PROCESSED / "capm_results.csv", index=False)
    print(f"[capm] -> {PROCESSED / 'capm_results.csv'}")
    return capm_df


# --------------------------------------------------------------------------- #
# Track B — Causal forest
# --------------------------------------------------------------------------- #

def build_cf_features(df: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Assemble feature matrix X for purchases from known parties."""
    purch = df[
        df["type"].eq("Purchase")
        & df["party"].isin(["Republican", "Democrat"])
        & df["excess_ret_90"].notna()
        & df["amount_mid_usd"].notna()
    ].copy()

    # Join market volatility on nearest trading date
    purch = purch.sort_values("transaction_date")
    vol_df = vol.reset_index().rename(columns={"date": "transaction_date"})
    purch = pd.merge_asof(purch, vol_df, on="transaction_date")

    purch["log_amount"] = np.log1p(purch["amount_mid_usd"])
    purch["day_of_week"] = purch["transaction_date"].dt.dayofweek
    purch["trade_year"] = purch["transaction_date"].dt.year

    # Sector dummies — keeps "Other" as the implicit base via drop_first
    sector_dummies = pd.get_dummies(purch["gics_sector"], prefix="sec", drop_first=True)

    X = pd.concat(
        [
            purch[["log_amount", "day_of_week", "trade_year", "market_vol_20d"]].reset_index(drop=True),
            sector_dummies.reset_index(drop=True),
        ],
        axis=1,
    ).fillna(0)

    return purch.reset_index(drop=True), X


def run_causal_forest(df: pd.DataFrame, vol: pd.Series) -> None:
    """Fit CausalForestDML. Treatment = Democrat (1) vs Republican (0)."""
    purch, X = build_cf_features(df, vol)

    Y = purch["excess_ret_90"].values
    T = (purch["party"] == "Democrat").astype(int).values
    X_arr = X.values.astype(float)

    print(f"[cf] n={len(Y)} | treated(D)={T.sum()} | control(R)={len(T)-T.sum()}")
    print(f"[cf] features: {list(X.columns)}")

    # Double-ML: partial out confounders before estimating heterogeneous effects.
    # GBM for Y residualisation; logistic for T (binary treatment) residualisation.
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
        model_t=LogisticRegression(max_iter=1000, C=0.1),
        discrete_treatment=True,
        n_estimators=2000,
        min_samples_leaf=20,
        max_features="auto",
        random_state=42,
        cv=3,
        verbose=0,
    )
    cf.fit(Y, T, X=X_arr)

    # Per-trade CATE estimates with 95% CI
    cate = cf.effect(X_arr)
    cate_lb, cate_ub = cf.effect_interval(X_arr, alpha=0.05)
    purch["cate"] = cate
    purch["cate_lb"] = cate_lb
    purch["cate_ub"] = cate_ub
    purch[["senator", "party", "ticker", "gics_sector", "transaction_date",
           "excess_ret_90", "cate", "cate_lb", "cate_ub"]].to_csv(
        PROCESSED / "cate_per_trade.csv", index=False
    )

    # Feature importance
    fi = pd.Series(cf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        PROCESSED / "cate_feature_importance.csv", index=False
    )
    ate_val = cf.ate(X_arr)
    ate_scalar = float(ate_val) if np.ndim(ate_val) == 0 else float(ate_val[0])
    print(f"[cf] ATE = {ate_scalar:+.4f}")
    print(f"[cf] feature importances:\n{fi.round(4).to_string()}")

    # Per-party CATE summary
    for party, flag in [("Democrat", 1), ("Republican", 0)]:
        mask = T == flag
        print(f"[cf] CATE {party}: mean={cate[mask].mean():+.4f}  "
              f"median={np.median(cate[mask]):+.4f}")

    return purch, X, cf, fi


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig1_cumulative_returns(df: pd.DataFrame) -> None:
    """Quarterly mean 90d excess return, cumulated over time, by party."""
    purch = df[
        df["type"].eq("Purchase")
        & df["party"].isin(["Republican", "Democrat"])
        & df["excess_ret_90"].notna()
    ].copy()
    purch["quarter"] = purch["transaction_date"].dt.to_period("Q")

    qmeans = (
        purch.groupby(["quarter", "party"])["excess_ret_90"]
        .mean()
        .reset_index()
    )
    qmeans["quarter_dt"] = qmeans["quarter"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 5))
    for party, grp in qmeans.groupby("party"):
        grp = grp.sort_values("quarter_dt")
        ax.plot(
            grp["quarter_dt"],
            grp["excess_ret_90"].cumsum(),
            label=party,
            color=PARTY_COLOURS[party],
            linewidth=2,
        )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative mean 90-day excess return on senator purchases")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative excess return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_cumulative_returns.png", dpi=150)
    plt.close(fig)
    print("[fig] fig1_cumulative_returns.png")


def fig2_capm_alpha(capm_df: pd.DataFrame) -> None:
    """Jensen's alpha by party and horizon with 95% CI error bars."""
    capm_df = capm_df.copy()
    capm_df["ci95"] = 1.96 * capm_df["alpha_se"]
    capm_df["horizon_label"] = capm_df["horizon_days"].astype(str) + "d"

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(HORIZONS))
    width = 0.35

    for i, party in enumerate(["Republican", "Democrat"]):
        sub = capm_df[capm_df["party"] == party].sort_values("horizon_days")
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            sub["alpha"],
            width,
            label=party,
            color=PARTY_COLOURS[party],
            alpha=0.85,
        )
        ax.errorbar(
            x + offset,
            sub["alpha"],
            yerr=sub["ci95"],
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.2,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in HORIZONS])
    ax.set_title("CAPM Jensen's alpha by party and horizon (HC3 SEs, ±95% CI)")
    ax.set_xlabel("Forward-return horizon")
    ax.set_ylabel("Jensen's alpha (annualised excess return)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_capm_alpha.png", dpi=150)
    plt.close(fig)
    print("[fig] fig2_capm_alpha.png")


def fig3_cate_distribution(purch_cf: pd.DataFrame) -> None:
    """Violin plot of per-trade CATEs by party."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=purch_cf,
        x="party",
        y="cate",
        hue="party",
        palette=PARTY_COLOURS,
        inner="quartile",
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Distribution of estimated CATEs by party\n"
                 "(treatment = Democratic purchase vs Republican purchase, 90d horizon)")
    ax.set_xlabel("")
    ax.set_ylabel("Estimated CATE (excess return differential)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_cate_distribution.png", dpi=150)
    plt.close(fig)
    print("[fig] fig3_cate_distribution.png")


def fig4_cate_by_sector(purch_cf: pd.DataFrame) -> None:
    """Mean CATE ± SE by GICS sector, horizontal bar."""
    stats = (
        purch_cf.groupby("gics_sector")["cate"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    stats = stats[stats["count"] >= 10]  # drop tiny sectors

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#d6604d" if m < 0 else "#2166ac" for m in stats["mean"]]
    ax.barh(stats["gics_sector"], stats["mean"], xerr=1.96 * stats["sem"],
            color=colours, alpha=0.85, capsize=4)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Mean CATE by GICS sector (±95% CI)\n"
                 "Positive = Democrats outperform Republicans in this sector")
    ax.set_xlabel("Mean CATE (90d excess return differential)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_cate_by_sector.png", dpi=150)
    plt.close(fig)
    print("[fig] fig4_cate_by_sector.png")


def fig5_sector_returns(df: pd.DataFrame) -> None:
    """Mean 90d excess return by sector × party (grouped bar)."""
    purch = df[
        df["type"].eq("Purchase")
        & df["party"].isin(["Republican", "Democrat"])
        & df["excess_ret_90"].notna()
        & ~df["gics_sector"].eq("Other")
    ]
    pivot = (
        purch.groupby(["gics_sector", "party"])["excess_ret_90"]
        .mean()
        .unstack("party")
        .sort_values("Democrat", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax,
               color=[PARTY_COLOURS["Democrat"], PARTY_COLOURS["Republican"]],
               alpha=0.85, width=0.7)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Mean 90-day excess return on purchases by sector and party")
    ax.set_xlabel("")
    ax.set_ylabel("Mean 90d excess return")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Party")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_sector_returns.png", dpi=150)
    plt.close(fig)
    print("[fig] fig5_sector_returns.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    df, bench = load_data()
    vol = market_volatility(bench)

    # Track A — CAPM
    print("\n=== CAPM ===")
    capm_df = run_capm(df)

    # Track B — Causal forest
    print("\n=== Causal Forest ===")
    purch_cf, X_feat, cf_model, fi = run_causal_forest(df, vol)

    # Figures
    print("\n=== Figures ===")
    fig1_cumulative_returns(df)
    fig2_capm_alpha(capm_df)
    fig3_cate_distribution(purch_cf)
    fig4_cate_by_sector(purch_cf)
    fig5_sector_returns(df)

    print("\nDone. All outputs in data/processed/")


if __name__ == "__main__":
    main()
