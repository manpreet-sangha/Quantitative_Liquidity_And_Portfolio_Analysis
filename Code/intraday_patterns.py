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
    ("abs_ret", "Mean |midquote return| (volatility)"),
]


def by_minute_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Average spread, depth, volume and minute volatility by stock and minute-of-day."""
    c = config.COLS
    return (df.groupby([c["stock"], "minute_of_day"])
              .agg(spread_bps=("spread_bps", "mean"),
                   depth_gbp=("depth_gbp", "mean"),
                   volume=(c["volume"], "mean"),
                   abs_ret=("abs_ret", "mean"))
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


def plot_volume_allocation(agg: pd.DataFrame, bin_minutes: int = 15):
    """Normalised intraday volume profile: each time-of-day bucket's share (%) of the
    stock's daily volume. This is the execution-schedule view a PM uses to slice a
    VWAP/TWAP order. Minutes are grouped into `bin_minutes` buckets so the underlying
    shape is legible (per-minute volume is too bursty to read)."""
    plot_style.apply_style()
    c = config.COLS
    fig, ax = plt.subplots(figsize=(7, 4))
    for stock, g in agg.groupby(c["stock"]):
        bucket = (g["minute_of_day"] // bin_minutes) * bin_minutes
        vol_by_bucket = g["volume"].groupby(bucket).sum()
        share = vol_by_bucket / vol_by_bucket.sum() * 100.0
        ax.plot(share.index / 60.0, share.values, linewidth=1.5, marker="o", markersize=3,
                label=config.STOCK_NAMES.get(stock, stock))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of daily volume (%)")
    ax.margins(x=0.01)
    ax.legend()
    path = plot_style.save_fig(fig, config.FIGURE_DIR, "intraday_volalloc")
    plt.close(fig)
    return path
