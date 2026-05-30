"""Random stock selection for Part 1 (O1 helper).

Picks one stock from each market-cap bucket (Large / Mid / Small) in the stock
universe. Kept in its own module so the selection rule is isolated from the rest
of the pipeline. Reproducible via a seed.
"""

import pandas as pd


def select_one_per_bucket(universe_csv, buckets, seed,
                          bucket_col="Size bucket", ticker_col="Stock", name_col="Name"):
    """Randomly select one stock from each market-cap bucket.

    Returns (stocks, names): `stocks` is a list of tickers ordered as `buckets`,
    `names` maps ticker -> company name. The draw is reproducible for a given seed.
    """
    df = pd.read_csv(universe_csv)
    stocks, names = [], {}
    for i, bucket in enumerate(buckets):
        pool = df[df[bucket_col] == bucket]
        if pool.empty:
            raise ValueError(f"No stocks in bucket '{bucket}' in {universe_csv}")
        pick = pool.sample(n=1, random_state=seed + i).iloc[0]
        ticker = pick[ticker_col]
        stocks.append(ticker)
        names[ticker] = pick[name_col] if name_col in df.columns else ticker
    return stocks, names
