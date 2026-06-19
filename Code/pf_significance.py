"""Part 2 - significance tests and higher moments for the momentum portfolios.

The basic summary table gives means, volatilities and Sharpe ratios, but it does not
say whether the long-short (HML) return is really different from zero, or how lop-sided
and fat-tailed the monthly returns are. This module adds two things:

  * Newey-West t-statistics for the HML average return and its market alpha. Newey-West
    simply means the t-stat allows for the returns being a little related from one month
    to the next, so it is not fooled into looking stronger than it is.
  * Skewness and kurtosis for every portfolio, which describe the shape of the returns
    (negative skew means the crashes are bigger than the rallies; high kurtosis means
    fat tails, i.e. extreme months happen more often than a normal bell curve would say).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _nw_lags(n: int) -> int:
    """A standard rule of thumb for how many months of memory to allow for."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def mean_tstat(returns: pd.Series):
    """Newey-West t-stat and p-value for the average monthly return being non-zero."""
    r = returns.dropna()
    fit = sm.OLS(r.values, np.ones((len(r), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(r))})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def alpha_tstat(returns: pd.Series, world: pd.Series):
    """Market alpha (the part of the return not explained by the world market) with a
    Newey-West t-stat. Returns the monthly alpha, its t-stat and its p-value."""
    df = pd.concat([returns, world.reindex(returns.index)], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = sm.add_constant(df.iloc[:, 1].values)        # column of 1s plus the world return
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(df))})
    return float(fit.params[0]), float(fit.tvalues[0]), float(fit.pvalues[0])


def higher_moments(portfolios: pd.DataFrame) -> pd.DataFrame:
    """Skewness and excess kurtosis of each portfolio's monthly returns (0 = normal)."""
    return pd.DataFrame({
        "Skew": portfolios.skew(),
        "Kurtosis": portfolios.kurt(),
    })
