"""Part 2 - data loading: turn the index levels into monthly returns.

The spreadsheet holds month-end index levels for a set of countries (already adjusted
for dividends, so a simple percentage change is the total return). This module loads
those levels, converts them to monthly returns, and adds a "World" column equal to the
equal-weighted average return across all countries each month.
"""

import pandas as pd
import pf_config


def load_levels() -> pd.DataFrame:
    """Load the monthly index levels with the date as the row index."""
    levels = pd.read_excel(pf_config.DATA_XLSX, sheet_name=0,
                           index_col=0, parse_dates=True)
    levels.index.name = "Date"
    return levels


def monthly_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Percentage change month to month for each country (drops the first row)."""
    return levels.pct_change().dropna(how="all")


def add_world(returns: pd.DataFrame) -> pd.DataFrame:
    """Append the equal-weighted average return across countries as 'World'."""
    out = returns.copy()
    out[pf_config.WORLD] = returns.mean(axis=1)
    return out


def load_returns() -> pd.DataFrame:
    """Convenience: levels -> monthly returns with the World column added."""
    return add_world(monthly_returns(load_levels()))


def country_columns(returns: pd.DataFrame) -> list:
    """The country names (everything except the World column)."""
    return [c for c in returns.columns if c != pf_config.WORLD]
