"""Part 1: summarise average liquidity per stock.

Produces a per-stock table of mean and median spread, mean depth, mean volume and
trade counts for the cross-sectional comparison of the three stocks.
"""

import pandas as pd
import config


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stock with the headline average-liquidity statistics."""
    c = config.COLS
    n_days = df.groupby(c["stock"])["date"].nunique()

    rows = []
    for stock, g in df.groupby(c["stock"]):
        rows.append({
            "Stock": stock,
            "Name": config.STOCK_NAMES.get(stock, stock),
            "Mean spread (bps)": g["spread_bps"].mean(),
            "Median spread (bps)": g["spread_bps"].median(),
            "Mean depth (GBP)": g["depth_gbp"].mean(),
            "Mean volume (sh/min)": g[c["volume"]].mean(),
            "Mean trades/min": g[c["num_trades"]].mean(),
            "ADV (sh/day)": g[c["volume"]].sum() / n_days[stock],
            "Half-spread (bps)": g["spread_bps"].mean() / 2,
            "Obs": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("Mean depth (GBP)", ascending=False).reset_index(drop=True)
