"""Streamlit app for SMM921 Part 1: Liquidity Analysis.

The input data is preloaded from the repo. On the front page the user chooses the
sample with two buttons (the default ULVR/EXPN/KGF, or a random one-per-market-cap
draw whose randomness is decided in code). Everything below - the tables, the
animated intraday GIFs, and the real top-of-book order book - is rebuilt for the
chosen stocks, so every legend and value stays in sync.

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
import liquidity_volatility
import intraday_gif
import lob_realdata

# The LOB visualiser pulls in Tk on import elsewhere; keep a headless backend.
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

st.set_page_config(page_title="SMM921 Liquidity Analysis", layout="wide")

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
    "Mean volume (sh/min)": "{:,.0f}",
    "Mean trades/min": "{:.1f}",
    "ADV (sh/day)": "{:,.0f}",
    "Half-spread (bps)": "{:.2f}",
    "Obs": "{:,.0f}",
    "Beta (vol)": "{:.4g}",
    "t-stat": "{:.2f}",
    "p-value": "{:.3g}",
    "R2": "{:.3f}",
    "Correlation": "{:.3f}",
    "N days": "{:.0f}",
}


def show_table(df):
    """Render a DataFrame as a large, high-contrast HTML table (theme-independent)."""
    fmt = {c: f for c, f in NUM_FMT.items() if c in df.columns}
    styler = df.style.hide(axis="index").format(fmt).set_table_styles([
        {"selector": "", "props": [("border-collapse", "collapse"), ("margin", "6px 0")]},
        {"selector": "th", "props": [("border", "1px solid #bbb"), ("font-size", "17px"),
                                     ("padding", "8px 14px"), ("background-color", "#1f4e79"),
                                     ("color", "#ffffff"), ("text-align", "center")]},
        {"selector": "td", "props": [("border", "1px solid #ccc"), ("font-size", "17px"),
                                     ("padding", "8px 14px"), ("color", "#111111"),
                                     ("background-color", "#ffffff"), ("text-align", "center")]},
    ])
    st.markdown(styler.to_html(), unsafe_allow_html=True)


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
def lob_real_gifs(stocks_key):
    df = cleaned_panel(stocks_key)
    out = {}
    for stock, g in df.groupby(config.COLS["stock"]):
        name = config.STOCK_NAMES.get(stock, stock)
        path = lob_realdata.OUT / f"lob_real_{stock.replace('.', '_')}.gif"
        lob_realdata.animate_stock(stock, g, name, path)
        out[stock] = str(path)
    return out


# ── Page + selection state ───────────────────────────────────────────
st.title("SMM921 — Liquidity Analysis (Part 1)")
info_text("Intraday liquidity of three London Stock Exchange stocks. "
          "Pick the sample on the left.")

universe = load_universe()
cap_info = universe.set_index("Stock")[["Market cap (GBP bn)", "Size bucket"]]

ss = st.session_state
ss.setdefault("mode", "default")
ss.setdefault("rseed", 0)
ss.setdefault("busy", True)

with st.sidebar:
    st.header("Stock selection")
    clicked_default = st.button(
        "Default (Unilever, Experian, Kingfisher)",
        disabled=ss.busy or ss.mode == "default",
        use_container_width=True)
    clicked_random = st.button(
        "Random (one per market-cap bucket)",
        disabled=ss.busy or ss.mode == "random",
        use_container_width=True)

# A switch enters the busy state so both buttons render disabled while processing.
if clicked_default:
    ss.mode = "default"; ss.busy = True; st.rerun()
if clicked_random:
    ss.mode = "random"; ss.rseed += 1; ss.busy = True; st.rerun()

# Resolve the selection (random seed is chosen in code, not from the front end).
if ss.mode == "default":
    stocks, names = list(DEFAULT_STOCKS), dict(DEFAULT_NAMES)
else:
    stocks, names = select_stocks.select_one_per_bucket(
        config.UNIVERSE_CSV, config.BUCKETS, ss.rseed)
set_selection(stocks, names)
key = tuple(stocks)

with st.sidebar:
    st.markdown("**Selected:**")
    for s in stocks:
        if s in cap_info.index:
            cap = cap_info.loc[s, "Market cap (GBP bn)"]
            bucket = cap_info.loc[s, "Size bucket"]
            extra = f", £{cap:.1f}bn, {bucket} cap"
        else:
            extra = ""
        st.write(f"- {names.get(s, s)} ({s}{extra})")

sel = (universe[universe["Stock"].isin(stocks)]
       [["Stock", "Name", "Market cap (GBP bn)", "Size bucket"]]
       .reset_index(drop=True))
st.subheader("Selected stocks")
show_table(sel)

# While switching, keep both buttons disabled until everything is rebuilt.
if ss.busy:
    with st.spinner("Processing selection, please wait…"):
        average_table(key)
        volatility_outputs(key)
        intraday_gifs(key)
        lob_real_gifs(key)
    ss.busy = False
    st.rerun()

# ── Results (all cached, instant) ────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Average liquidity", "Intraday patterns",
     "Liquidity vs volatility", "Order book (real data)"])

with tab1:
    st.subheader("Average liquidity over the sample")
    show_table(average_table(key))

with tab2:
    gpaths = intraday_gifs(key)
    st.subheader("Animated intraday patterns")
    c1, c2, c3 = st.columns(3)
    c1.image(gpaths[0])
    c2.image(gpaths[1])
    c3.image(gpaths[2])

with tab3:
    reg, sfigs = volatility_outputs(key)
    s1, s2 = st.columns(2)
    s1.image(sfigs[0], use_container_width=True)
    s2.image(sfigs[1], use_container_width=True)
    st.subheader("Regression of daily liquidity on daily volatility")
    show_table(reg)

with tab4:
    info_text("Real top-of-book from the input data for each stock: best bid, "
              "best ask, their sizes, the mid and the spread, minute by minute.")
    lpaths = lob_real_gifs(key)
    for stock, gif in lpaths.items():
        st.markdown(f"**{config.STOCK_NAMES.get(stock, stock)} ({stock})**")
        st.image(gif)
