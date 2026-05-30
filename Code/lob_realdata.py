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

    spread_p = ask - bid
    half_range = max(spread_p.max(), 0.02) * 0.75   # fixed x half-range (pence from mid)
    vmax = max(bsize.max(), asize.max(), 1.0) * 1.50

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)

    def update(k):
        ax.clear()
        ax.set_facecolor(BG)
        s = max(spread_p[k], 1e-9)
        xb, xa = -s / 2.0, s / 2.0
        bw = max(half_range * 0.14, s * 0.30)
        # Bid and ask placed by their distance from the mid (x=0), so the gap
        # between them is the spread. Only the bars move; the side labels do not.
        ax.bar(xb, bsize[k], width=bw, color=BID_COLOUR, edgecolor="darkorange", linewidth=0.8)
        ax.bar(xa, asize[k], width=bw, color=ASK_COLOUR, edgecolor="steelblue", linewidth=0.8)
        ax.text(xb, bsize[k], f"{bid[k]:.2f}p\n{bsize[k]:,.0f} sh",
                ha="center", va="bottom", fontsize=10)
        ax.text(xa, asize[k], f"{ask[k]:.2f}p\n{asize[k]:,.0f} sh",
                ha="center", va="bottom", fontsize=10)
        # Mid line, fixed at the centre.
        ax.axvline(0, color=MID_COLOUR, linestyle="--", linewidth=1.4)
        ax.text(0, vmax * 0.02, "Mid", color="#1e8c50", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        # Spread as a horizontal double-headed arrow between best bid and best ask.
        y_arrow = vmax * 0.80
        ax.annotate("", xy=(xa, y_arrow), xytext=(xb, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.8))
        ax.text(0, y_arrow + vmax * 0.02,
                f"Spread  {spread_bps[k]:.2f} bps  ({s:.2f}p)",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey"))
        # Fixed side labels (never move).
        ax.text(-half_range, vmax * 0.99, "Best Bid", color="darkorange",
                fontsize=12, fontweight="bold", ha="left", va="top")
        ax.text(half_range, vmax * 0.99, "Best Ask", color="steelblue",
                fontsize=12, fontweight="bold", ha="right", va="top")
        ax.set_xlim(-half_range, half_range)
        ax.set_ylim(0, vmax)
        ax.set_xlabel("Distance from mid (pence)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Shares available at best price", fontsize=12, fontweight="bold")
        ax.set_title(f"{name} ({stock}) — real top-of-book", fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=10)
        # Date and time banner (fixed position, updates each frame).
        ax.text(0.5, 0.94, stamps[k], transform=ax.transAxes, ha="center", va="top",
                fontsize=12, fontweight="bold", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="grey"))

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
