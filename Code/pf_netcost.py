"""Part 2 - net-of-cost performance of the optimised portfolios.

Our recommendation to prefer the robust covariance rests on turnover: the sample
portfolio trades far more each month than the robust one, and trading is not free. This
module subtracts a simple trading cost from each month's return and recomputes the
Sharpe ratio, so we can see whether the gross edge survives once costs are paid.

Net return in a month = gross return minus (cost per unit traded) times that month's
turnover, where turnover is how much of the book was bought and sold (the sum of the
absolute weight changes). Costs are quoted in basis points, where 1 bp = 0.01%.
"""

import numpy as np
import pandas as pd

import pf_config
import pf_optimize

M = pf_config.MONTHS_PER_YEAR


def net_returns(gross: pd.Series, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Gross monthly returns minus the cost of that month's turnover."""
    cost = cost_bps / 1e4                                     # basis points -> fraction
    traded = pf_optimize.turnover(weights).reindex(gross.index).fillna(0.0)
    return gross - cost * traded


def net_sharpe(gross: pd.Series, weights: pd.DataFrame, cost_bps: float) -> float:
    """Annualised Sharpe ratio of the net-of-cost returns."""
    net = net_returns(gross, weights, cost_bps)
    ann_mean = net.mean() * M
    ann_vol = net.std() * np.sqrt(M)
    return (ann_mean - pf_config.RISK_FREE) / ann_vol


def cost_table(gross_by_method: dict, weights_by_method: dict,
               cost_levels_bps) -> pd.DataFrame:
    """Net-of-cost annualised Sharpe for each method at each assumed cost level."""
    rows = []
    for name in gross_by_method:
        row = {"Portfolio": name,
               "Gross Sharpe": net_sharpe(gross_by_method[name],
                                          weights_by_method[name], 0.0)}
        for c in cost_levels_bps:
            row[f"Net Sharpe @ {c}bps"] = net_sharpe(
                gross_by_method[name], weights_by_method[name], c)
        rows.append(row)
    return pd.DataFrame(rows)
