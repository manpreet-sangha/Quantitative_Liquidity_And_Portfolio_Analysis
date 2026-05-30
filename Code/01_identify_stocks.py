"""
SMM921 Coursework - Part 1 (Liquidity Analysis): Stage 1, Step 1.

Objectives:
  1. Identify the stocks contained in the trading-data file.
  2. Obtain the market capitalisation of each stock (via yfinance) so that a
     three-stock sample spanning a range of market caps can be selected.

Note: market cap is not in the trading data, so it is pulled from yfinance.
The current value is used as a proxy for the early-2026 sample period.

Run:  venv/Scripts/python.exe SMM921_Coursework/Code/01_identify_stocks.py
"""

from pathlib import Path
import pandas as pd
import yfinance as yf
import config

INPUT_CSV = config.RAW_CSV
OUTPUT_CSV = config.OUTPUT_DIR / "stock_universe.csv"


def identify_stocks(csv_path: Path) -> list[str]:
    """Return the sorted list of unique stock identifiers in the data file."""
    return sorted(pd.read_csv(csv_path, usecols=["Stock"])["Stock"].unique())


def fetch_market_caps(tickers: list[str]) -> pd.DataFrame:
    """Fetch market cap, currency and last price for each ticker via yfinance.

    For LSE (.L) tickers, info['marketCap'] is reported in GBP even though the
    price quote currency is GBp (pence).
    """
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        rows.append({
            "Stock": t,
            "Name": info.get("longName") or info.get("shortName"),
            "Market cap": info.get("marketCap"),
            "Currency": info.get("currency"),
            "Last price": info.get("regularMarketPrice"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    tickers = identify_stocks(INPUT_CSV)
    print(f"{len(tickers)} stocks identified:\n{', '.join(tickers)}\n")

    caps = fetch_market_caps(tickers)
    caps["Market cap (GBP bn)"] = caps["Market cap"] / 1e9
    caps = caps.sort_values("Market cap", ascending=False).reset_index(drop=True)
    caps["Size bucket"] = pd.qcut(caps["Market cap"], 3, labels=["Small", "Mid", "Large"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    caps.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(caps[["Stock", "Name", "Market cap (GBP bn)", "Currency", "Size bucket"]].to_string(index=False))
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
