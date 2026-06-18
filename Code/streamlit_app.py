"""Streamlit app for SMM921: Trading and Market Microstructure.

Two top-level tabs:
  * Liquidity Analysis (Part 1) - intraday liquidity of three chosen LSE stocks. Under
    "Average liquidity" we show the average-liquidity table, then the intraday-pattern
    animations, then the real-data order-book animations per stock. A second sub-tab
    relates daily liquidity to daily volatility.
  * Portfolio Analysis (Part 2) - a 34-country equity portfolio (independent of the
    Part 1 stock choice): performance and beta, momentum-sorted portfolios and HML, and
    the sample-vs-robust covariance mean-variance optimisation.

Run:  venv/Scripts/streamlit run SMM921_Coursework/Code/streamlit_app.py
"""

import sys
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import config
import select_stocks
import data_loader
import liquidity_measures
import cleaning
import average_liquidity
import liquidity_volatility
import intraday_gif
import intraday_patterns
import lob_realdata
import pf_config
import pf_data
import pf_performance
import pf_momentum
import pf_optimize
import pf_correlation

# Streamlit Cloud sometimes hot-reloads this script on a git update without re-importing
# changed modules, leaving a stale pf_correlation in sys.modules (missing newly added
# helpers such as country_returns). Force a refresh so the current module is always used
# without needing a manual app reboot.
importlib.reload(pf_correlation)

# The LOB visualiser pulls in Tk on import elsewhere; keep a headless backend.
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

st.set_page_config(page_title="SMM921 Trading and Market Microstructure", layout="wide")

DEFAULT_STOCKS = config.CONFIG["part1"]["stocks"]
DEFAULT_NAMES = config.CONFIG["part1"]["stock_names"]


def info_text(text):
    """Render a clearly visible note (larger than the faint st.caption style)."""
    st.markdown(
        f"<p style='font-size:1.15rem; line-height:1.5; color:#222;'>{text}</p>",
        unsafe_allow_html=True,
    )


NUM_FMT = {
    "Market cap (GBP bn)": "{:.1f}",
    "Mean spread (bps)": "{:.2f}",
    "Median spread (bps)": "{:.2f}",
    "Mean depth (GBP)": "{:,.0f}",
    "Turnover (GBP/min)": "{:,.0f}",
    "Value ADV (GBP/day)": "{:,.0f}",
    "Amihud illiq (x1e6)": "{:.3f}",
    "Mean volume (sh/min)": "{:,.0f}",
    "Mean trades/min": "{:.1f}",
    "ADV (sh/day)": "{:,.0f}",
    "Half-spread (bps)": "{:.2f}",
    "Obs": "{:,.0f}",
    # Regression table: show complete (fixed-point) numbers, not exponential.
    "Beta (vol)": "{:,.2f}",
    "t-stat": "{:.2f}",
    "p-value": "{:.6f}",
    "R2": "{:.3f}",
    "Correlation": "{:.3f}",
    "N days": "{:.0f}",
    # Part 2 tables.
    "Ann. mean (%)": "{:.2f}",
    "Ann. vol (%)": "{:.2f}",
    "Sharpe": "{:.3f}",
    "Beta": "{:.3f}",
    "Ann. alpha (%)": "{:.2f}",
    "Info ratio": "{:.3f}",
    "Max drawdown (%)": "{:.2f}",
}


def show_table(df):
    """Render a DataFrame as a large, high-contrast HTML table (theme-independent)."""
    fmt = {c: f for c, f in NUM_FMT.items() if c in df.columns}
    styler = df.style.hide(axis="index").format(fmt, na_rep="—").set_table_styles([
        {"selector": "", "props": [("border-collapse", "collapse"), ("margin", "6px 0")]},
        {"selector": "th", "props": [("border", "1px solid #bbb"), ("font-size", "17px"),
                                     ("padding", "8px 14px"), ("background-color", "#1f4e79"),
                                     ("color", "#ffffff"), ("text-align", "center")]},
        {"selector": "td", "props": [("border", "1px solid #ccc"), ("font-size", "17px"),
                                     ("padding", "8px 14px"), ("color", "#111111"),
                                     ("background-color", "#ffffff"), ("text-align", "center")]},
    ])
    st.markdown(styler.to_html(), unsafe_allow_html=True)


# Figures are shown centred at a comfortable fraction of the page. Full-width is too big
# in the wide layout and 3-up columns were too small. To resize every plot at once, change
# the middle number of FIG_COLS: bigger middle = bigger plots (e.g. (1, 3, 1) ~= 60%).
FIG_COLS = (1, 2, 1)   # centred, middle column ~= 50% of the page width


def show_fig(path, caption=None, align="left"):
    """Render a figure/animation at a medium size, left-justified by default.

    align='left' (default) or 'center'."""
    if align == "left":
        box = st.columns([FIG_COLS[1], FIG_COLS[0] + FIG_COLS[2]])[0]
    else:
        box = st.columns(FIG_COLS)[1]
    if caption:
        box.markdown(caption)
    box.image(path, width="stretch")


def set_selection(stocks, names):
    """Point the Part 1 pipeline at the chosen stocks (drives every legend)."""
    config.PART1_STOCKS = list(stocks)
    config.STOCK_NAMES = dict(names)


@st.cache_data(show_spinner=False)
def load_universe():
    return pd.read_csv(config.UNIVERSE_CSV)


@st.cache_data(show_spinner=False)
def cleaned_panel(stocks_key):
    df = data_loader.load_part1()
    df = liquidity_measures.add_measures(df)
    df, _ = cleaning.apply_filters(df)
    return df


@st.cache_data(show_spinner=False)
def average_table(stocks_key):
    return average_liquidity.summarise(cleaned_panel(stocks_key))


@st.cache_data(show_spinner=False)
def volatility_outputs(stocks_key):
    daily = liquidity_volatility.daily_panel(cleaned_panel(stocks_key))
    reg = liquidity_volatility.regress(daily)
    figs = [str(p) for p in liquidity_volatility.plot_scatter(daily)]
    return reg, figs


@st.cache_data(show_spinner=False)
def intraday_gifs(stocks_key):
    intraday_gif.main()
    return [str(intraday_gif.OUT / f"intraday_{m}.gif")
            for m in ("spread", "depth", "volume")]


@st.cache_data(show_spinner=False)
def intraday_extra_figs(stocks_key):
    """Static intraday volatility and volume-allocation profiles (Part 1 augmentation)."""
    agg = intraday_patterns.by_minute_of_day(cleaned_panel(stocks_key))
    paths = [str(p) for p in intraday_patterns.plot_intraday(agg)]
    volalloc = str(intraday_patterns.plot_volume_allocation(agg))
    volatility = next(p for p in paths if "abs_ret" in p)
    return [volatility, volalloc]


@st.cache_data(show_spinner=False)
def lob_real_gifs(stocks_key):
    df = cleaned_panel(stocks_key)
    out = {}
    for stock, g in df.groupby(config.COLS["stock"]):
        name = config.STOCK_NAMES.get(stock, stock)
        path = lob_realdata.OUT / f"lob_real_{stock.replace('.', '_')}.gif"
        lob_realdata.animate_stock(stock, g, name, path)
        out[stock] = str(path)
    return out


@st.cache_data(show_spinner=False)
def portfolio_outputs():
    """Part 2 - build the 34-country portfolio tables and figures once (cached).

    Independent of the Part 1 stock selection, so it takes no arguments.
    """
    r = pf_data.load_returns()
    cs = pf_data.country_columns(r)
    world = r[pf_config.WORLD]
    figs = {}

    def _path(result):
        """Keep the saved figure path and close the figure to free memory at once."""
        path, fig = result
        plt.close(fig)
        return str(path)

    perf = pf_performance.summarise(r)
    figs["rr"] = _path(pf_performance.plot_risk_return(perf))
    figs["beta"] = _path(pf_performance.plot_beta(perf))

    signal = pf_momentum.momentum_signal(r[cs])
    ports = pf_momentum.sorted_portfolios(r[cs], signal)
    panel = ports.copy()
    panel[pf_config.WORLD] = world.reindex(ports.index)
    mstats = pf_momentum.portfolio_stats(panel, world)
    figs["mcum"] = _path(pf_momentum.plot_cumulative(ports))
    figs["mono"] = _path(pf_momentum.plot_monotonicity(mstats))
    figs["hml"] = _path(pf_momentum.plot_hml(ports))

    s_ret, s_w = pf_optimize.run(r, signal, robust=False)
    rb_ret, r_w = pf_optimize.run(r, signal, robust=True)
    ostats = pf_optimize.compare_stats(s_ret, rb_ret, world)
    figs["ocum"] = _path(pf_optimize.plot_cumulative(s_ret, rb_ret, world))
    figs["olev"] = _path(pf_optimize.plot_leverage(s_w, r_w))
    figs["oturn"] = _path(pf_optimize.plot_turnover(s_w, r_w))

    return perf, mstats, ostats, figs


@st.cache_data(show_spinner=False)
def correlation_data():
    """Part 2 - country returns plus the list of selectable 60-month window-end dates."""
    r = pf_data.load_returns()
    _, R = pf_correlation.country_returns(r)
    return r, pf_correlation.window_end_dates(R)


@st.cache_data(show_spinner=False)
def correlation_gif(version):
    """Part 2 - rolling 60-month correlation heatmap GIF.

    `version` keys the cache on the plot's look (colormap, size, frames) so changing the
    style regenerates the GIF instead of serving a stale cached file. Streamlit keeps the
    cache across hot-reloads, so a no-argument version would never refresh after an edit.
    """
    r = pf_data.load_returns()
    out = pf_correlation.OUT / "corr_rolling.gif"
    pf_correlation.animate(r, out)
    return str(out)


# ── Page + selection state ───────────────────────────────────────────
st.title("SMM921 — Trading and Market Microstructure")
info_text("Part 1 (Liquidity Analysis) studies the intraday liquidity of three London "
          "Stock Exchange stocks; pick the sample on the left. Part 2 (Portfolio "
          "Analysis) studies a 34-country equity portfolio using all countries, so it "
          "does not depend on that choice.")

universe = load_universe()
cap_info = universe.set_index("Stock")[["Market cap (GBP bn)", "Size bucket"]]

ss = st.session_state
ss.setdefault("mode", "default")
ss.setdefault("rseed", 0)
ss.setdefault("busy", True)

with st.sidebar:
    st.header("Stock selection (Part 1)")
    clicked_default = st.button(
        "Default (Unilever, Experian, Kingfisher)",
        disabled=ss.busy or ss.mode == "default",
        width="stretch")
    clicked_random = st.button(
        "Random (one per market-cap bucket)",
        disabled=ss.busy,
        width="stretch")
    info_text("Applies to the Liquidity Analysis tab only.")

if clicked_default:
    ss.mode = "default"; ss.busy = True; st.rerun()
if clicked_random:
    ss.mode = "random"; ss.rseed += 1; ss.busy = True; st.rerun()

if ss.mode == "default":
    stocks, names = list(DEFAULT_STOCKS), dict(DEFAULT_NAMES)
else:
    stocks, names = select_stocks.select_one_per_bucket(
        config.UNIVERSE_CSV, config.BUCKETS, ss.rseed)
set_selection(stocks, names)
key = tuple(stocks)

with st.sidebar:
    st.markdown("**Selected (Part 1):**")
    for s in stocks:
        if s in cap_info.index:
            cap = cap_info.loc[s, "Market cap (GBP bn)"]
            bucket = cap_info.loc[s, "Size bucket"]
            extra = f", £{cap:.1f}bn, {bucket} cap"
        else:
            extra = ""
        st.write(f"- {names.get(s, s)} ({s}{extra})")

# Warm the Part 1 caches while a selection change is processing.
if ss.busy:
    with st.spinner("Processing selection, please wait…"):
        average_table(key)
        volatility_outputs(key)
        intraday_gifs(key)
        lob_real_gifs(key)
    ss.busy = False
    st.rerun()

tab_liq, tab_pf = st.tabs(["Liquidity Analysis", "Portfolio Analysis"])

# ── Part 1: Liquidity Analysis ───────────────────────────────────────
with tab_liq:
    sel = (universe[universe["Stock"].isin(stocks)]
           [["Stock", "Name", "Market cap (GBP bn)", "Size bucket"]]
           .reset_index(drop=True))
    st.subheader("Selected stocks")
    show_table(sel)

    liq_avg, liq_intra, liq_vol, liq_ob = st.tabs(
        ["Average liquidity", "Intraday patterns",
         "Liquidity vs volatility", "Order book (real data)"])

    with liq_avg:
        st.subheader("Average liquidity over the sample")
        show_table(average_table(key))

    with liq_intra:
        st.subheader("Intraday liquidity patterns")
        for gif in intraday_gifs(key):
            show_fig(gif, align="left")
        st.subheader("Intraday volatility and volume allocation")
        for fig in intraday_extra_figs(key):
            show_fig(fig, align="left")

    with liq_vol:
        reg, sfigs = volatility_outputs(key)
        for fig in sfigs:
            show_fig(fig)
        st.subheader("Regression of daily liquidity on daily volatility")
        show_table(reg)

    with liq_ob:
        st.subheader("Order book (real data) by stock")
        for stock, gif in lob_real_gifs(key).items():
            show_fig(gif, caption=f"**{config.STOCK_NAMES.get(stock, stock)} ({stock})**",
                     align="left")

# ── Part 2: Portfolio Analysis ───────────────────────────────────────
with tab_pf:
    with st.spinner("Building the portfolio analysis (34 countries)…"):
        perf, mstats, ostats, pfigs = portfolio_outputs()
    info_text("A 34-country equity portfolio over 20 years of monthly data "
              "(independent of the Part 1 stock choice).")

    pf_perf, pf_mom, pf_opt, pf_corr = st.tabs(
        ["Performance & risk", "Momentum portfolios", "Mean-variance optimisation",
         "Rolling correlations"])

    with pf_perf:
        st.subheader("Performance and systematic risk by country")
        show_table(perf)
        show_fig(pfigs["rr"])
        show_fig(pfigs["beta"])

    with pf_mom:
        st.subheader("Momentum-sorted portfolios and HML")
        show_table(mstats)
        show_fig(pfigs["mcum"])
        show_fig(pfigs["mono"])
        show_fig(pfigs["hml"])

    with pf_opt:
        st.subheader("Sample versus robust covariance")
        show_table(ostats)
        show_fig(pfigs["ocum"])
        show_fig(pfigs["olev"])
        show_fig(pfigs["oturn"])

    with pf_corr:
        st.subheader("Rolling 60-month return correlations")
        info_text("The optimiser estimates its covariance matrix from the most recent 60 "
                  "months of returns. Drag the slider to move that 60-month window through "
                  "time. Cells shift from blue toward red as cross-country correlations rise "
                  "and diversification evaporates, most dramatically in the 2008 and 2020 "
                  "crises, then cool again afterwards.")
        r_corr, dates = correlation_data()
        labels = [d.strftime("%Y-%m") for d in dates]
        sel = st.select_slider("Window ending (month)", options=labels, value=labels[-1])
        end_date = dates[labels.index(sel)]
        fig = pf_correlation.heatmap_figure(r_corr, end_date)
        # Pin the save dpi: other Part 2 plots call plot_style.apply_style(), which sets a
        # high global savefig.dpi that would otherwise bloat this figure.
        with matplotlib.rc_context({"savefig.dpi": 110}):
            st.columns([5, 1])[0].pyplot(fig)
        plt.close(fig)
        if st.checkbox("Play the full sweep as an animation (builds a GIF)"):
            gif_version = (f"{pf_correlation.CMAP}-{pf_correlation.FIGSIZE}-"
                           f"{pf_correlation.TARGET_FRAMES}-{pf_correlation.FPS}")
            with st.spinner("Building the animation…"):
                st.columns([5, 1])[0].image(correlation_gif(gif_version), width="stretch")

        st.subheader("What a portfolio manager and trader take from this")
        st.markdown(
            "- **Diversification is regime-dependent.** Correlations jump toward 1 in 2008 and "
            "2020, so a calm-period diversified book becomes one global-equity bet in a crash. "
            "Budget risk against crisis correlations and cut gross exposure as the grid reddens.\n"
            "- **Regional clusters are redundant.** The developed-Europe bloc sits near 0.9, so "
            "holding several of those markets is really one position. Durable diversification "
            "comes from the persistently loose names (Brazil, India, Indonesia, Korea), though "
            "they too converge in a global crisis.\n"
            "- **Size and leverage inversely to the correlation regime.** The same positions are "
            "riskier when the grid is red, so scale gross exposure down as average correlation "
            "rises (correlation-adjusted volatility targeting).\n"
            "- **Hedge from outside equities.** Because intra-equity correlations converge in "
            "stress, crisis protection must come from index puts, volatility, Treasuries, gold or "
            "trend following rather than from another country.\n"
            "- **The matrix is unstable, so the optimiser needs a robust covariance.** The "
            "window-to-window movement seen here is the estimation noise that makes a "
            "sample-covariance optimiser over-fit; it is why the robust constant-correlation "
            "version in the Mean-variance optimisation tab is far better behaved."
        )
        st.caption(
            "Caveats: these are reasonable inferences from one 2006–2026 sample of monthly "
            "country-index returns on 60-month windows. The levels are sample-specific and the "
            "long window smooths and lags regime shifts (for trading signals you would use a "
            "shorter window). They are sound as risk-management and construction principles, not "
            "a backtested trading system."
        )
