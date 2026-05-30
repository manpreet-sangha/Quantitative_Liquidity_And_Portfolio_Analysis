"""Streamlit app for SMM921 Part 1: Liquidity Analysis.

The input data is preloaded from the repo. On the front page the user chooses the
sample (the default ULVR/EXPN/KGF, or a random one-per-market-cap-bucket draw). The
app then shows the average-liquidity table, intraday and volatility plots, and the
animated GIFs (intraday spread/depth and the calibrated order book simulation),
all regenerated for whatever stocks are selected so every legend stays in sync.

Run:  venv/Scripts/streamlit run SMM921_Coursework/Code/streamlit_app.py
"""

import sys
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
import intraday_patterns
import liquidity_volatility
import intraday_gif
import lob_simulation

# Keep a headless backend: importing the LOB visualiser pulls in Tk, which crashes
# under Streamlit's worker threads.
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

st.set_page_config(page_title="SMM921 Liquidity Analysis", layout="wide")

DEFAULT_STOCKS = config.CONFIG["part1"]["stocks"]
DEFAULT_NAMES = config.CONFIG["part1"]["stock_names"]


def set_selection(stocks, names):
    """Point the whole pipeline at the chosen stocks (drives every legend)."""
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
def intraday_figures(stocks_key):
    agg = intraday_patterns.by_minute_of_day(cleaned_panel(stocks_key))
    return [str(p) for p in intraday_patterns.plot_intraday(agg)]


@st.cache_data(show_spinner=False)
def volatility_outputs(stocks_key):
    daily = liquidity_volatility.daily_panel(cleaned_panel(stocks_key))
    reg = liquidity_volatility.regress(daily)
    figs = [str(p) for p in liquidity_volatility.plot_scatter(daily)]
    return reg, figs


@st.cache_data(show_spinner=False)
def intraday_gifs(stocks_key):
    intraday_gif.main()
    return [str(intraday_gif.OUT / "intraday_spread.gif"),
            str(intraday_gif.OUT / "intraday_depth.gif")]


@st.cache_data(show_spinner=False)
def lob_gifs(stocks_key):
    df = cleaned_panel(stocks_key)
    out = {}
    for stock, g in df.groupby(config.COLS["stock"]):
        name = config.STOCK_NAMES.get(stock, stock)
        _, _, gif = lob_simulation.run_stock(stock, g, name, seconds=5.0, fps=8)
        out[stock] = str(gif)
    return out


# ── Sidebar: selection ───────────────────────────────────────────────
st.title("SMM921 — Liquidity Analysis (Part 1)")
st.caption("Intraday liquidity of three London Stock Exchange stocks. "
           "Pick the sample on the left.")

universe = load_universe()

with st.sidebar:
    st.header("Stock selection")
    mode = st.radio(
        "Sample",
        ["Default (Unilever, Experian, Kingfisher)",
         "Random (one per market-cap bucket)"])
    if mode.startswith("Random"):
        seed = st.number_input("Random seed", min_value=0,
                               value=int(config.RANDOM_SEED), step=1)
        stocks, names = select_stocks.select_one_per_bucket(
            config.UNIVERSE_CSV, config.BUCKETS, int(seed))
    else:
        stocks, names = list(DEFAULT_STOCKS), dict(DEFAULT_NAMES)
    st.markdown("**Selected:**")
    for s in stocks:
        st.write(f"- {names.get(s, s)} ({s})")

set_selection(stocks, names)
key = tuple(stocks)

# ── Selected stocks + market caps ────────────────────────────────────
sel = (universe[universe["Stock"].isin(stocks)]
       [["Stock", "Name", "Market cap (GBP bn)", "Size bucket"]]
       .reset_index(drop=True))
st.subheader("Selected stocks")
st.dataframe(sel, use_container_width=True, hide_index=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Average liquidity", "Intraday patterns",
     "Liquidity vs volatility", "Order book simulation"])

with tab1:
    st.subheader("Average liquidity over the sample")
    st.dataframe(average_table(key), use_container_width=True, hide_index=True)

with tab2:
    figs = intraday_figures(key)
    c1, c2 = st.columns(2)
    c1.image(figs[0], use_container_width=True)
    c2.image(figs[1], use_container_width=True)
    st.image(figs[2], use_container_width=True)
    st.subheader("Animated intraday spread and depth")
    with st.spinner("Building intraday animations…"):
        gpaths = intraday_gifs(key)
    g1, g2 = st.columns(2)
    g1.image(gpaths[0])
    g2.image(gpaths[1])

with tab3:
    reg, sfigs = volatility_outputs(key)
    s1, s2 = st.columns(2)
    s1.image(sfigs[0], use_container_width=True)
    s2.image(sfigs[1], use_container_width=True)
    st.subheader("Regression of daily liquidity on daily volatility")
    st.dataframe(reg, use_container_width=True, hide_index=True)

with tab4:
    st.caption("Limit order book simulation calibrated to each stock's real "
               "midquote, spread and depth from the input data.")
    with st.spinner("Building order book animations…"):
        lpaths = lob_gifs(key)
    cols = st.columns(len(lpaths))
    for col, (stock, gif) in zip(cols, lpaths.items()):
        col.markdown(f"**{config.STOCK_NAMES.get(stock, stock)} ({stock})**")
        col.image(gif, use_container_width=True)
