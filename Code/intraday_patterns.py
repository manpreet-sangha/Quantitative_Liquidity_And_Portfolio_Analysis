"""Part 1: describe how liquidity varies across the trading day.

Averages spread, depth and volume across days by minute-of-day for each stock and
plots the three stocks together so the intraday pattern can be compared.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

MEASURES = [
    ("spread_bps", "Mean spread (bps)"),
    ("depth_gbp", "Mean depth (GBP)"),
    ("volume", "Mean volume (shares/min)"),
]


def by_minute_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Average spread, depth and volume by stock and minute-of-day."""
    c = config.COLS
    return (df.groupby([c["stock"], "minute_of_day"])
              .agg(spread_bps=("spread_bps", "mean"),
                   depth_gbp=("depth_gbp", "mean"),
                   volume=(c["volume"], "mean"))
              .reset_index())


def plot_intraday(agg: pd.DataFrame) -> list:
    """Save one figure per measure (stocks overlaid). Returns the file paths."""
    c = config.COLS
    paths = []
    for col, label in MEASURES:
        fig, ax = plt.subplots(figsize=(7, 5))
        for stock, g in agg.groupby(c["stock"]):
            ax.plot(g["minute_of_day"] / 60.0, g[col], linewidth=0.9,
                    label=config.STOCK_NAMES.get(stock, stock))
        ax.set_xlabel("Hour of day", fontsize=16)
        ax.set_ylabel(label, fontsize=16)
        ax.set_title(f"Intraday pattern: {label}", fontsize=17, fontweight="bold")
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=15, framealpha=0.9)
        path = config.FIGURE_DIR / f"intraday_{col}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
