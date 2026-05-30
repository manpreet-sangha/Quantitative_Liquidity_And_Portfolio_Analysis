"""Part 1: relate daily liquidity to daily volatility.

Daily volatility is defined as the mean of the absolute minutely midquote returns
within a day. For each stock this regresses daily spread (and daily depth) on daily
volatility using heteroskedasticity-robust standard errors, and reports the
correlation.
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

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
    """Scatter of each daily liquidity measure vs daily volatility."""
    c = config.COLS
    paths = []
    for dep, label in DEPENDENTS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for stock, g in daily.groupby(c["stock"]):
            ax.scatter(g["daily_vol"], g[dep], s=30, alpha=0.8,
                       label=config.STOCK_NAMES.get(stock, stock))
        ax.set_xlabel("Daily volatility", fontsize=16)
        ax.set_ylabel(label, fontsize=16)
        ax.set_title(f"{label} vs daily volatility", fontsize=17, fontweight="bold")
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=15, markerscale=1.8, framealpha=0.9)
        path = config.FIGURE_DIR / f"vol_{dep}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
