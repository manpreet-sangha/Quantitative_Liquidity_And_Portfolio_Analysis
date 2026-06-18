"""Part 1: describe how liquidity varies across the trading day.

Averages spread, depth and volume across days by minute-of-day for each stock and
plots the three stocks together so the intraday pattern can be compared.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import plot_style

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
    """Save one figure per measure (stocks overlaid) as PDF + PNG. Returns the paths."""
    plot_style.apply_style()
    c = config.COLS
    paths = []
    for col, label in MEASURES:
        fig, ax = plt.subplots(figsize=(7, 4))
        for stock, g in agg.groupby(c["stock"]):
            ax.plot(g["minute_of_day"] / 60.0, g[col], linewidth=1.3,
                    label=config.STOCK_NAMES.get(stock, stock))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(label)
        ax.margins(x=0.01)
        ax.legend()
        paths.append(plot_style.save_fig(fig, config.FIGURE_DIR, f"intraday_{col}"))
        plt.close(fig)
    return paths
