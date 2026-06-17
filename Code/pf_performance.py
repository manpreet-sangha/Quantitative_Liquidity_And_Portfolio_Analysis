"""Part 2 - performance and market sensitivity per country.

For every country we work out the annualised average return, the annualised
volatility, the Sharpe ratio, and the beta to the world market (how much the country
tends to move when the world market moves). These let us compare which countries paid
off best for the risk taken and which were most exposed to global swings.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pf_config
import pf_data
import plot_style

M = pf_config.MONTHS_PER_YEAR


def summarise(returns: pd.DataFrame) -> pd.DataFrame:
    """One row per country: annualised mean, vol, Sharpe and beta to the world."""
    countries = pf_data.country_columns(returns)
    world = returns[pf_config.WORLD]
    world_var = world.var()

    rows = []
    for c in countries:
        r = returns[c]
        ann_mean = r.mean() * M
        ann_vol = r.std() * np.sqrt(M)
        sharpe = (ann_mean - pf_config.RISK_FREE) / ann_vol
        beta = r.cov(world) / world_var
        rows.append({
            "Country": c,
            "Ann. mean (%)": ann_mean * 100,
            "Ann. vol (%)": ann_vol * 100,
            "Sharpe": sharpe,
            "Beta": beta,
        })
    return (pd.DataFrame(rows)
            .sort_values("Sharpe", ascending=False)
            .reset_index(drop=True))


def extremes(summary: pd.DataFrame, column: str, n: int = 5) -> pd.DataFrame:
    """Top and bottom n countries on a given column (for the write-up)."""
    top = summary.nlargest(n, column).assign(Group=f"Top {n}")
    bottom = summary.nsmallest(n, column).assign(Group=f"Bottom {n}")
    return pd.concat([top, bottom]).reset_index(drop=True)


def plot_risk_return(summary: pd.DataFrame):
    """Scatter of annualised return against annualised volatility, one dot per country."""
    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(summary["Ann. vol (%)"], summary["Ann. mean (%)"],
               s=46, alpha=0.8, color=plot_style.PALETTE[0], edgecolors="white",
               linewidths=0.5)
    for _, row in summary.iterrows():
        ax.annotate(row["Country"], (row["Ann. vol (%)"], row["Ann. mean (%)"]),
                    fontsize=9, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised mean return (%)")
    ax.set_title("Risk and return by country")
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_risk_return"), fig


def plot_beta(summary: pd.DataFrame):
    """Bar chart of each country's beta to the world market, sorted high to low."""
    plot_style.apply_style()
    s = summary.sort_values("Beta", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(s["Country"], s["Beta"], color=plot_style.PALETTE[2], edgecolor="white")
    ax.axhline(1.0, color="0.3", linewidth=0.9, linestyle="--", label="Beta = 1")
    ax.set_ylabel("Beta to world market")
    ax.set_title("Sensitivity to the world market by country")
    ax.tick_params(axis="x", labelrotation=90, labelsize=9)
    ax.legend()
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_beta"), fig
