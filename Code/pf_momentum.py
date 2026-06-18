"""Part 2 - momentum signal and momentum-sorted portfolios.

The momentum signal for a country in a given month is its total return over the
prior twelve months, leaving out the most recent month (a one-month gap is the
standard way to avoid the short-term bounce-back effect). Each month we rank the
countries by this signal, split them into five equal groups, and hold each group over
the next month. HML buys the top group and sells the bottom group.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pf_config
import pf_data
import plot_style

M = pf_config.MONTHS_PER_YEAR


def momentum_signal(country_returns: pd.DataFrame) -> pd.DataFrame:
    """Total return over the look-back window, skipping the most recent month(s).

    With a 12-month look-back and a 1-month skip, the signal in month t is the
    compounded return over months t-12 to t-2 (eleven months).
    """
    gross = 1.0 + country_returns
    win = pf_config.MOM_LOOKBACK - pf_config.MOM_SKIP   # months actually compounded
    shift = pf_config.MOM_SKIP + 1                      # leave out the most recent month
    return gross.shift(shift).rolling(win).apply(np.prod, raw=True) - 1.0


def sorted_portfolios(country_returns: pd.DataFrame,
                      signal: pd.DataFrame) -> pd.DataFrame:
    """Five momentum-sorted portfolios (P1=losers … P5=winners) plus HML, monthly.

    Each month the countries are ranked by the signal and split into five equal
    groups; the group earns the equal-weighted average of its members' returns that
    month. HML = P5 return − P1 return.
    """
    n = pf_config.N_PORTFOLIOS
    records = {}
    for t in signal.index:
        s = signal.loc[t].dropna()
        if len(s) < n or t not in country_returns.index:
            continue
        held = country_returns.loc[t]
        groups = pd.qcut(s.rank(method="first"), n, labels=range(1, n + 1))
        row = {f"P{g}": held[groups[groups == g].index].mean() for g in range(1, n + 1)}
        row["HML"] = row[f"P{n}"] - row["P1"]
        records[t] = row
    return pd.DataFrame(records).T.sort_index().astype(float)


def _max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall of a £1 investment that compounds the returns."""
    cum = (1.0 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def portfolio_stats(portfolios: pd.DataFrame, world: pd.Series) -> pd.DataFrame:
    """Annualised performance and risk numbers for each portfolio column."""
    world = world.reindex(portfolios.index)
    world_var = world.var()
    rows = []
    for col in portfolios.columns:
        r = portfolios[col]
        ann_mean = r.mean() * M
        ann_vol = r.std() * np.sqrt(M)
        beta = r.cov(world) / world_var
        ann_alpha = (r.mean() - beta * world.mean()) * M
        resid_vol = (r - beta * world).std() * np.sqrt(M)
        rows.append({
            "Portfolio": col,
            "Ann. mean (%)": ann_mean * 100,
            "Ann. vol (%)": ann_vol * 100,
            "Sharpe": (ann_mean - pf_config.RISK_FREE) / ann_vol,
            "Beta": beta,
            "Ann. alpha (%)": ann_alpha * 100,
            "Info ratio": ann_alpha / resid_vol if resid_vol > 0 else np.nan,
            "Max drawdown (%)": _max_drawdown(r) * 100,
        })
    return pd.DataFrame(rows)


def plot_cumulative(portfolios: pd.DataFrame):
    """Growth of £1 for the five momentum-sorted portfolios (P1 to P5)."""
    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, col in enumerate([c for c in portfolios.columns if c.startswith("P")]):
        cum = (1.0 + portfolios[col]).cumprod()
        ax.plot(portfolios.index, cum, label=col,
                color=plot_style.PALETTE[i % len(plot_style.PALETTE)])
    ax.set_ylabel("Growth of £1")
    ax.set_xlabel("Date")
    ax.legend(ncol=5, fontsize=9)
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_mom_cumulative"), fig


def plot_monotonicity(stats: pd.DataFrame):
    """Bar chart of annualised mean return P1→P5 (should rise if momentum works)."""
    plot_style.apply_style()
    s = stats[stats["Portfolio"].str.startswith("P")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s["Portfolio"], s["Ann. mean (%)"], color=plot_style.PALETTE[2],
           edgecolor="white")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_ylabel("Annualised mean return (%)")
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_mom_monotonicity"), fig


def plot_hml(portfolios: pd.DataFrame):
    """Growth of £1 in the HML spread (winners minus losers), showing the momentum crash."""
    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    cum = (1.0 + portfolios["HML"]).cumprod()
    ax.plot(portfolios.index, cum, color=plot_style.PALETTE[1], label="HML (P5 - P1)")
    ax.axhline(1.0, color="0.4", linewidth=0.8)
    ax.set_ylabel("Growth of £1")
    ax.set_xlabel("Date")
    ax.legend()
    return plot_style.save_fig(fig, pf_config.FIGURE_DIR, "pf_mom_hml"), fig
