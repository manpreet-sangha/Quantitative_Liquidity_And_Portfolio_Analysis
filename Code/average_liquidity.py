"""Part 1: summarise average liquidity per stock.

Produces a per-stock table for the cross-sectional comparison of the three stocks.
It covers the three dimensions of liquidity: cost (bid-ask spread), quantity (depth
and GBP turnover) and price impact (the Amihud (2002) illiquidity ratio), plus trade
counts.
"""

import pandas as pd
import config


def _amihud(g: pd.DataFrame) -> float:
    """Amihud (2002) illiquidity for one stock: median across days of the daily mean of
    |midquote return| / GBP volume, over traded minutes only. Scaled by 1e6 for reading."""
    c = config.COLS
    t = g[(g[c["volume"]] > 0) & (g["gbp_volume"] > 0) & g["abs_ret"].notna()]
    if t.empty:
        return float("nan")
    daily = (t["abs_ret"] / t["gbp_volume"]).groupby(t["date"]).mean()
    return daily.median() * 1e6


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
            "Turnover (GBP/min)": g["gbp_volume"].mean(),
            "Value ADV (GBP/day)": g["gbp_volume"].sum() / n_days[stock],
            "Amihud illiq (x1e6)": _amihud(g),
            "Mean volume (sh/min)": g[c["volume"]].mean(),
            "Mean trades/min": g[c["num_trades"]].mean(),
            "ADV (sh/day)": g[c["volume"]].sum() / n_days[stock],
            "Half-spread (bps)": g["spread_bps"].mean() / 2,
            "Obs": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("Mean depth (GBP)", ascending=False).reset_index(drop=True)
