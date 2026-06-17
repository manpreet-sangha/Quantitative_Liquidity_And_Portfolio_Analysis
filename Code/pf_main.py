"""SMM921 Part 2 - Portfolio Analysis pipeline.

Runs the stages end to end: load data and build returns, compare country
performance and market sensitivity, test the momentum signal with sorted portfolios
and an HML spread, then run the mean-variance optimisation with both the sample and
the robust (constant-correlation) covariance. All tables and figures are written to
output/part2/.

Run:  venv/Scripts/python.exe SMM921_Coursework/Code/pf_main.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pf_config
import pf_data
import pf_performance
import pf_momentum
import pf_optimize

OUT = pf_config.OUTPUT_DIR


def _save(df, name, index=False):
    df.to_csv(OUT / name, index=index)


def main():
    # --- D: data + returns + world ---
    returns = pf_data.load_returns()
    countries = pf_data.country_columns(returns)
    print(f"Loaded {len(countries)} countries, {len(returns)} monthly returns "
          f"({returns.index[0].date()} to {returns.index[-1].date()})\n")
    _save(returns, "monthly_returns.csv", index=True)

    # --- E: performance & systematic risk ---
    perf = pf_performance.summarise(returns)
    _save(perf, "performance.csv")
    print("Performance & beta (top/bottom by Sharpe):")
    print(pf_performance.extremes(perf, "Sharpe").to_string(index=False), "\n")
    for fn in (pf_performance.plot_risk_return, pf_performance.plot_beta):
        _, fig = fn(perf); plt.close(fig)

    # --- F: momentum signal, sorted portfolios, HML ---
    cret = returns[countries]
    signal = pf_momentum.momentum_signal(cret)
    ports = pf_momentum.sorted_portfolios(cret, signal)
    # Add the world market as a benchmark row in the stats (not in the plots).
    mom_panel = ports.copy()
    mom_panel[pf_config.WORLD] = returns[pf_config.WORLD].reindex(ports.index)
    pstats = pf_momentum.portfolio_stats(mom_panel, returns[pf_config.WORLD])
    _save(ports, "momentum_portfolios.csv", index=True)
    _save(pstats, "momentum_stats.csv")
    print(f"Momentum portfolios: {len(ports)} months "
          f"({ports.index[0].date()} to {ports.index[-1].date()})")
    print(pstats.round(3).to_string(index=False), "\n")
    _, fig = pf_momentum.plot_cumulative(ports); plt.close(fig)
    _, fig = pf_momentum.plot_monotonicity(pstats); plt.close(fig)
    _, fig = pf_momentum.plot_hml(ports); plt.close(fig)

    # --- G + H: mean-variance optimisation (sample vs robust covariance) ---
    s_ret, s_w = pf_optimize.run(returns, signal, robust=False)
    r_ret, r_w = pf_optimize.run(returns, signal, robust=True)
    ostats = pf_optimize.compare_stats(s_ret, r_ret, returns[pf_config.WORLD])
    _save(s_ret.to_frame(), "opt_returns_sample.csv", index=True)
    _save(r_ret.to_frame(), "opt_returns_robust.csv", index=True)
    _save(ostats, "opt_stats.csv")
    print(f"Optimised portfolios: {len(s_ret)} months "
          f"({s_ret.index[0].date()} to {s_ret.index[-1].date()})")
    print(ostats.round(3).to_string(index=False))
    print(f"Mean gross leverage  - sample {pf_optimize.gross_leverage(s_w).mean():.2f}, "
          f"robust {pf_optimize.gross_leverage(r_w).mean():.2f}")
    print(f"Mean monthly turnover - sample {pf_optimize.turnover(s_w).mean():.2f}, "
          f"robust {pf_optimize.turnover(r_w).mean():.2f}")
    _, fig = pf_optimize.plot_cumulative(s_ret, r_ret, returns[pf_config.WORLD]); plt.close(fig)
    _, fig = pf_optimize.plot_leverage(s_w, r_w); plt.close(fig)
    _, fig = pf_optimize.plot_turnover(s_w, r_w); plt.close(fig)

    print(f"\nAll Part 2 outputs written to {OUT}")


if __name__ == "__main__":
    main()
