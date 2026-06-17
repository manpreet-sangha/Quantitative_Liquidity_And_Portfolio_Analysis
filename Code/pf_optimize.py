"""Part 2 - mean-variance optimisation driven by the momentum signal.

Each month, from the most recent 60 months of data, we build:
  * an expected-return ("alpha") for each country from the momentum signal,
  * a covariance matrix of country returns,
and then choose the fully-invested weights that best trade off return against risk.
We do this two ways for comparison: with the ordinary sample covariance, and with a
steadier "constant-correlation" version that keeps each country's own variance but
replaces every pairwise correlation with the average one. The weights are held over
the following month to give a realised return series.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pf_config
import pf_data
import pf_momentum
import plot_style

IC = pf_config.INFO_COEFF
LAM = pf_config.RISK_AVERSION
WIN = pf_config.WINDOW


def _mv_weights(alpha: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Fully-invested mean-variance weights (they sum to 1).

    Closed form w = inv(cov)(alpha - gamma*1)/lam, where gamma is set so the weights
    add up to one.
    """
    inv = np.linalg.pinv(cov)
    ones = np.ones(len(alpha))
    inv_ones = inv @ ones
    gamma = (alpha @ inv_ones - lam) / (ones @ inv_ones)
    return inv @ (alpha - gamma * ones) / lam


def _constant_correlation(cov: np.ndarray) -> np.ndarray:
    """Keep the variances, replace every correlation with the average correlation."""
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    corr = cov / outer
    n = len(cov)
    mean_corr = (corr.sum() - n) / (n * (n - 1))   # average of the off-diagonal terms
    robust = np.full((n, n), mean_corr)
    np.fill_diagonal(robust, 1.0)
    return robust * outer


def run(returns: pd.DataFrame, signal: pd.DataFrame, robust: bool = False):
    """Walk forward month by month and return (realised returns, weights table).

    Starting once 60 months of history exist, weights are formed from the trailing
    window and held over the next month.
    """
    countries = pf_data.country_columns(returns)
    R = returns[countries]
    idx = R.index
    realised, weights = {}, {}

    for i in range(WIN, len(idx)):
        window = R.iloc[i - WIN:i]                 # most recent 60 months
        sig = signal.iloc[i]                       # momentum known this month
        if sig.isna().any():
            continue
        z = (sig - sig.mean()) / sig.std()         # standardise across countries
        resid_vol = window.std().mean()            # average country volatility (scalar)
        alpha = (IC * resid_vol * z).values        # signal turned into expected return
        cov = window.cov().values
        if robust:
            cov = _constant_correlation(cov)
        w = _mv_weights(alpha, cov, LAM)
        realised[idx[i]] = float(w @ R.iloc[i].values)   # held over the next month
        weights[idx[i]] = w

    name = "Robust" if robust else "Sample"
    ret = pd.Series(realised, name=name).sort_index()
    wdf = pd.DataFrame(weights, index=countries).T.sort_index()
    return ret, wdf


def compare_stats(sample_ret: pd.Series, robust_ret: pd.Series,
                  world: pd.Series) -> pd.DataFrame:
    """Side-by-side performance/risk table for the two optimised series and the world."""
    panel = pd.DataFrame({
        "Sample": sample_ret,
        "Robust": robust_ret,
        pf_config.WORLD: world.reindex(sample_ret.index),
    })
    return pf_momentum.portfolio_stats(panel, world)


def gross_leverage(weights: pd.DataFrame) -> pd.Series:
    """Sum of the absolute weights each month (1 means no leverage / no shorting)."""
    return weights.abs().sum(axis=1)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """How much the weights change month to month (sum of absolute changes)."""
    return weights.diff().abs().sum(axis=1).dropna()


def plot_cumulative(sample_ret, robust_ret, world):
    """Growth of £1 for the two optimised portfolios against the world market."""
    plot_style.apply_style()
    w = world.reindex(sample_ret.index)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sample_ret.index, (1 + sample_ret).cumprod(), label="Sample cov.",
            color=plot_style.PALETTE[0])
    ax.plot(robust_ret.index, (1 + robust_ret).cumprod(), label="Robust cov.",
            color=plot_style.PALETTE[1])
    ax.plot(w.index, (1 + w).cumprod(), label="World (equal-weight)",
            color="0.5", linewidth=1.2, linestyle="--")
    ax.set_ylabel("Growth of \\pounds1")
    ax.set_xlabel("Date")
    ax.set_title("Optimised portfolios vs the world market")
    ax.legend()
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_opt_cumulative"), fig


def plot_leverage(sample_w, robust_w):
    """Gross leverage over time for the two covariance methods."""
    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sample_w.index, gross_leverage(sample_w), label="Sample cov.",
            color=plot_style.PALETTE[0])
    ax.plot(robust_w.index, gross_leverage(robust_w), label="Robust cov.",
            color=plot_style.PALETTE[1])
    ax.axhline(1.0, color="0.3", linewidth=0.9, linestyle="--", label="No leverage")
    ax.set_ylabel("Sum of absolute weights")
    ax.set_xlabel("Date")
    ax.set_title("Gross leverage of the optimised portfolios")
    ax.legend()
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_opt_leverage"), fig


def plot_turnover(sample_w, robust_w):
    """Month-to-month turnover (how much the weights move) for the two methods."""
    plot_style.apply_style()
    ts, tr = turnover(sample_w), turnover(robust_w)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts.index, ts, color=plot_style.PALETTE[0],
            label=f"Sample cov. (mean {ts.mean():.1f})")
    ax.plot(tr.index, tr, color=plot_style.PALETTE[1],
            label=f"Robust cov. (mean {tr.mean():.1f})")
    ax.set_ylabel("Turnover (sum of absolute weight changes)")
    ax.set_xlabel("Date")
    ax.set_title("Monthly turnover of the optimised portfolios")
    ax.legend()
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_opt_turnover"), fig
