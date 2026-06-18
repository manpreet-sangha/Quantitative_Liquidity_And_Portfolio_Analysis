"""SMM921 Part 1 - Liquidity Analysis pipeline.

Orchestrates the per-objective modules end to end. All parameters come from
config.json (stocks, paths, filters), so this file contains no hard-coded values.

Run:  venv/Scripts/python.exe SMM921_Coursework/Code/main.py
"""

import json

import config
import data_loader
import liquidity_measures
import cleaning
import average_liquidity
import intraday_patterns
import liquidity_volatility


def main():
    print(f"Selection mode : {config.SELECTION_MODE}")
    print(f"Part 1 stocks  : {', '.join(config.PART1_STOCKS)}\n")

    # Load -> measures (O2) -> clean/filter (O3)
    df = data_loader.load_part1()
    df = liquidity_measures.add_measures(df)
    df, report = cleaning.apply_filters(df)
    df.to_csv(config.OUTPUT_DIR / "liquidity_minute.csv", index=False)
    print(f"Cleaned panel  : {len(df):,} rows "
          f"({report['n_initial']:,} -> {report['n_after_outliers']:,})")
    print(f"Outliers removed: {report['outliers_removed_per_stock']}\n")

    # O4 - average liquidity
    summary = average_liquidity.summarise(df)
    summary.to_csv(config.OUTPUT_DIR / "avg_liquidity.csv", index=False)
    print("Average liquidity (O4):")
    print(summary.to_string(index=False), "\n")

    # O5 - intraday patterns
    agg = intraday_patterns.by_minute_of_day(df)
    agg.to_csv(config.OUTPUT_DIR / "intraday_patterns.csv", index=False)
    intraday_patterns.plot_intraday(agg)
    intraday_patterns.plot_volume_allocation(agg)

    # O6 - liquidity vs volatility
    daily = liquidity_volatility.daily_panel(df)
    daily.to_csv(config.OUTPUT_DIR / "daily_panel.csv", index=False)
    reg = liquidity_volatility.regress(daily)
    reg.to_csv(config.OUTPUT_DIR / "liquidity_vs_vol.csv", index=False)
    liquidity_volatility.plot_scatter(daily)
    print("Liquidity vs volatility (O6):")
    print(reg.to_string(index=False), "\n")

    with open(config.OUTPUT_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"All outputs written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
