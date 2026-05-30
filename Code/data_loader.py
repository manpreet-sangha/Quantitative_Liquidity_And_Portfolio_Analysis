"""Part 1 - data loading: prepare the raw 1-minute data for the analysis.

Restricts the dataset to the selected Part 1 stocks, parses timestamps, and
applies the no-trade-minute convention: a blank Last/Volume/Num. Trades means no
trade occurred that minute, so traded volume and trade count are set to zero.
"""

import pandas as pd
import config


def load_part1() -> pd.DataFrame:
    """Return the raw per-minute data for the Part 1 stocks only."""
    c = config.COLS
    df = pd.read_csv(config.RAW_CSV)
    df = df[df[c["stock"]].isin(config.PART1_STOCKS)].copy()

    df[c["datetime"]] = pd.to_datetime(df[c["datetime"]])

    # No-trade minutes: no trade occurred -> zero traded volume / trade count.
    df[c["volume"]] = df[c["volume"]].fillna(0)
    df[c["num_trades"]] = df[c["num_trades"]].fillna(0)

    df["date"] = df[c["datetime"]].dt.date
    df["minute_of_day"] = df[c["datetime"]].dt.hour * 60 + df[c["datetime"]].dt.minute

    return df.sort_values([c["stock"], c["datetime"]]).reset_index(drop=True)
