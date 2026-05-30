"""Part 1: real top-of-book animation per stock, driven entirely by the input data.

For each selected stock this animates the ACTUAL best bid, best ask, their sizes,
the midquote and the spread over a window of real one-minute observations from the
input file. Every value shown comes from the input data, with no simulation. One
frame every 0.25 seconds. Config-driven, so it works for any selected stocks.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import config
import data_loader
import liquidity_measures
import cleaning

OUT = config.OUTPUT_DIR / "lob_real"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 4            # one frame every 0.25 seconds
N_FRAMES = 80      # minutes shown across the chosen day

BID_COLOUR = "#F5A623"   # orange (buy side)
ASK_COLOUR = "#4A90D9"   # blue (sell side)
MID_COLOUR = "#2ECC71"   # green (mid)
BG = "#FAFAFA"


def _panel():
    df = data_loader.load_part1()
    df = liquidity_measures.add_measures(df)
    df, _ = cleaning.apply_filters(df)
    return df


def _sample_window(g):
    """Sample ~N_FRAMES observations evenly across the whole sample period.

    Spanning the full period (not a single day) means the date as well as the time
    advances as the animation plays.
    """
    c = config.COLS
    d = g.sort_values(c["datetime"]).reset_index(drop=True)
    if len(d) > N_FRAMES:
        idx = np.linspace(0, len(d) - 1, N_FRAMES).astype(int)
        d = d.iloc[idx].reset_index(drop=True)
    return d


def animate_stock(stock, g, name, path):
    """Animate one stock's real best bid/ask, sizes, mid and spread over a day."""
    c = config.COLS
    d = _sample_window(g)
    bid = d[c["bid"]].to_numpy(float)
    ask = d[c["ask"]].to_numpy(float)
    bsize = d[c["bid_size"]].to_numpy(float)
    asize = d[c["ask_size"]].to_numpy(float)
    mid = d["mid"].to_numpy(float)
    spread_bps = d["spread_bps"].to_numpy(float)
    stamps = d[c["datetime"]].dt.strftime("%Y-%m-%d  %H:%M").to_numpy()

    vmax = max(bsize.max(), asize.max(), 1.0) * 1.20

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)

    def update(k):
        ax.clear()
        ax.set_facecolor(BG)
        # Fixed positions: the Best Bid bar sits at x=0 and the Best Ask bar at
        # x=1, so their labels never move. Only the heights (sizes) and the
        # displayed prices, mid, spread and date/time change each frame.
        ax.bar(0, bsize[k], width=0.6, color=BID_COLOUR,
               edgecolor="darkorange", linewidth=0.8)
        ax.bar(1, asize[k], width=0.6, color=ASK_COLOUR,
               edgecolor="steelblue", linewidth=0.8)
        ax.text(0, bsize[k], f"{bsize[k]:,.0f} sh", ha="center", va="bottom", fontsize=11)
        ax.text(1, asize[k], f"{asize[k]:,.0f} sh", ha="center", va="bottom", fontsize=11)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, vmax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Best Bid\n{bid[k]:.2f}p", f"Best Ask\n{ask[k]:.2f}p"],
                           fontsize=12, fontweight="bold")
        ax.set_ylabel("Shares available at best price", fontsize=13, fontweight="bold")
        ax.set_title(f"{name} ({stock}) — real top-of-book", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)
        ax.text(0.5, 0.97,
                f"{stamps[k]}\nMid {mid[k]:.2f}p     Spread {spread_bps[k]:.2f} bps",
                transform=ax.transAxes, ha="center", va="top", fontsize=12,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey"))

    anim = animation.FuncAnimation(fig, update, frames=len(d),
                                   interval=1000 // FPS, blit=False)
    anim.save(str(path), writer="pillow", fps=FPS)
    plt.close(fig)
    return path


def main():
    df = _panel()
    for stock, g in df.groupby(config.COLS["stock"]):
        name = config.STOCK_NAMES.get(stock, stock)
        animate_stock(stock, g, name,
                      OUT / f"lob_real_{stock.replace('.', '_')}.gif")
    print(f"Saved real top-of-book GIFs for "
          f"{', '.join(config.PART1_STOCKS)} to {OUT}")


if __name__ == "__main__":
    main()
