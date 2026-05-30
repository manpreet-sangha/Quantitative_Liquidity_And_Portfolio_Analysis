"""Part 1: clean and filter the per-minute data.

Keeps only the continuous trading session, drops non-positive quotes and negative
spreads, and removes per-stock spread outliers. Returns the cleaned frame plus a
report of how many rows were removed at each step.
"""

import pandas as pd
import config


def apply_filters(df: pd.DataFrame):
    """Filter to the trading session, drop bad quotes, and remove spread outliers."""
    c = config.COLS
    start, end = config.session_bounds_min()
    report = {"n_initial": int(len(df))}

    # 1) Trading session only (e.g. 08:15-16:25) -> excludes auctions/open/close.
    df = df[(df["minute_of_day"] >= start) & (df["minute_of_day"] <= end)].copy()
    report["n_after_session"] = int(len(df))

    # 2) Drop non-positive quotes and negative spreads (data errors).
    df = df[(df[c["bid"]] > 0) & (df[c["ask"]] > 0)]
    if config.FILTERS.get("drop_negative_spread", True):
        df = df[df["spread_bps"] >= 0]
    report["n_after_spread"] = int(len(df))

    # 3) Per-stock upper-tail outlier removal on spread_bps (IQR rule).
    k = config.FILTERS.get("outlier_iqr_multiple", 5.0)
    kept, removed = [], {}
    for stock, g in df.groupby(c["stock"]):
        q1, q3 = g["spread_bps"].quantile([0.25, 0.75])
        upper = q3 + k * (q3 - q1)
        mask = g["spread_bps"] <= upper
        removed[stock] = int((~mask).sum())
        kept.append(g[mask])
    df = pd.concat(kept).sort_values([c["stock"], c["datetime"]]).reset_index(drop=True)

    report["n_after_outliers"] = int(len(df))
    report["outlier_rule"] = f"spread_bps > Q3 + {k} * IQR (per stock)"
    report["outliers_removed_per_stock"] = removed
    return df, report
