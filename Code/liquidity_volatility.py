"""Part 1: relate daily liquidity to daily volatility.

Daily volatility is defined as the mean of the absolute minutely midquote returns
within a day. For each stock this regresses daily spread (and daily depth) on daily
volatility using heteroskedasticity-robust standard errors, and reports the
correlation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import plot_style

DEPENDENTS = [
    ("daily_spread", "Daily mean spread (bps)"),
    ("daily_depth", "Daily mean depth (GBP)"),
]


def daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the minute panel to one row per stock-day."""
    c = config.COLS
    return (df.groupby([c["stock"], "date"])
              .agg(daily_vol=("ret", lambda r: r.abs().mean()),
                   daily_spread=("spread_bps", "mean"),
                   daily_depth=("depth_gbp", "mean"))
              .reset_index())


def _ols(y, x):
    return sm.OLS(y, sm.add_constant(x), missing="drop").fit(cov_type="HC0")


def regress(daily: pd.DataFrame) -> pd.DataFrame:
    """OLS of each daily liquidity measure on daily volatility, per stock."""
    c = config.COLS
    rows = []
    for stock, g in daily.groupby(c["stock"]):
        for dep, _ in DEPENDENTS:
            m = _ols(g[dep], g["daily_vol"])
            rows.append({
                "Stock": stock,
                "Name": config.STOCK_NAMES.get(stock, stock),
                "Dependent": dep,
                "Beta (vol)": m.params["daily_vol"],
                "t-stat": m.tvalues["daily_vol"],
                "p-value": m.pvalues["daily_vol"],
                "R2": m.rsquared,
                "Correlation": g[[dep, "daily_vol"]].corr().iloc[0, 1],
                "N days": int(m.nobs),
            })
    return pd.DataFrame(rows)


def plot_scatter(daily: pd.DataFrame) -> list:
    """Scatter of each daily liquidity measure vs daily volatility (PDF + PNG)."""
    plot_style.apply_style()
    c = config.COLS
    paths = []
    for dep, label in DEPENDENTS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (stock, g) in enumerate(daily.groupby(c["stock"])):
            color = plot_style.PALETTE[i % len(plot_style.PALETTE)]
            ax.scatter(g["daily_vol"], g[dep], s=28, alpha=0.7,
                       edgecolors="white", linewidths=0.3, color=color,
                       label=config.STOCK_NAMES.get(stock, stock))
            # Fitted OLS line (same slope as the regression) to visualise beta.
            b1, b0 = np.polyfit(g["daily_vol"], g[dep], 1)
            xx = np.linspace(g["daily_vol"].min(), g["daily_vol"].max(), 50)
            ax.plot(xx, b0 + b1 * xx, color=color, linewidth=2.2)
        ax.set_xlabel("Daily volatility")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs daily volatility")
        ax.legend(markerscale=1.6)
        paths.append(plot_style.save_fig(fig, config.FIGURE_DIR, f"vol_{dep}"))
        plt.close(fig)
    return paths
