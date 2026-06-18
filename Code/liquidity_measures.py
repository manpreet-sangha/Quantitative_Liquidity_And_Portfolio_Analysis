"""Part 1: build the per-minute liquidity measures.

For each stock-minute this computes the midquote, the midquote return, the bid-ask
spread in basis points, and the depth in GBP. Bid/Ask are quoted in pence, so the
GBP depth is divided by the configured pence->GBP divisor; the spread in bps is
unit-free.
"""

import pandas as pd
import config

BPS = 10_000  # basis-points scaling (definition of "in bps")


def add_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid, spread_bps, depth_gbp and (within-day) midquote returns."""
    c = config.COLS
    out = df.copy()
    bid, ask = out[c["bid"]], out[c["ask"]]

    out["mid"] = 0.5 * (bid + ask)
    out["spread_bps"] = (ask - bid) / out["mid"] * BPS
    out["depth_gbp"] = (0.5 * (out[c["bid_size"]] * bid + out[c["ask_size"]] * ask)
                        / config.PRICE_DIVISOR)

    # Midquote returns within each stock-day (avoids overnight jumps).
    out["ret"] = out.groupby([c["stock"], "date"])["mid"].pct_change()
    out["abs_ret"] = out["ret"].abs()

    # GBP value traded per minute (shares x last price in pence -> GBP). No-trade
    # minutes have no Last, so the value is 0. Used for GBP turnover (cross-stock
    # comparable activity) and the Amihud (2002) illiquidity ratio.
    out["gbp_volume"] = (out[c["volume"]] * out[c["last"]] / config.PRICE_DIVISOR).fillna(0.0)
    return out
