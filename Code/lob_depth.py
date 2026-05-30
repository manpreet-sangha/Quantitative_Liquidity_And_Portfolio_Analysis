"""Part 1: top-of-book depth-and-spread visualisation for the selected stocks.

Adapts the market-depth visualisation from the author's limit order book
simulation project (https://github.com/manpreet-sangha/limit_order_book_simulation)
to real LSE data: bid volume is drawn upward (orange) and ask volume downward
(blue), with a green mid line and a shaded bid-ask spread.

Because real spreads are tiny relative to price, the x-axis is the distance from
the midquote in basis points (not absolute price), so the spread is visible and
directly comparable across stocks. Bar heights are the depth in GBP. The stocks
shown are whatever config.PART1_STOCKS contains, so this adapts automatically if
the three stocks are re-selected (fixed or random).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_loader
import liquidity_measures
import cleaning

# Visual palette adapted from the LOB simulation project.
BID_COLOUR = "#F5A623"   # orange (buy side)
ASK_COLOUR = "#4A90D9"   # blue (sell side)
MID_COLOUR = "#2ECC71"   # green (mid price)
BG_COLOUR = "#FAFAFA"


def cleaned_panel():
    """Load, build measures and clean the per-minute panel for the chosen stocks."""
    df = data_loader.load_part1()
    df = liquidity_measures.add_measures(df)
    df, _ = cleaning.apply_filters(df)
    return df


def snapshot(g):
    """Typical top-of-book snapshot for one stock: average spread and GBP depth."""
    c = config.COLS
    div = config.PRICE_DIVISOR
    return {
        "spread_bps": g["spread_bps"].mean(),
        "bid_depth_gbp": (g[c["bid_size"]] * g[c["bid"]] / div).mean(),
        "ask_depth_gbp": (g[c["ask_size"]] * g[c["ask"]] / div).mean(),
        "mid_price": g["mid"].mean() / div,   # in pounds
    }


def _draw(ax, snap, name):
    """Draw one stock's top-of-book depth chart on the given axis."""
    half = snap["spread_bps"] / 2.0
    width = snap["spread_bps"] * 0.45 or 0.5

    ax.bar(-half, snap["bid_depth_gbp"], width=width, color=BID_COLOUR,
           edgecolor="darkorange", linewidth=0.8, label="Bid (buy)", zorder=3)
    ax.bar(half, snap["ask_depth_gbp"], width=width, color=ASK_COLOUR,
           edgecolor="steelblue", linewidth=0.8, label="Ask (sell)", zorder=3)

    ax.axvline(0, color=MID_COLOUR, linestyle="--", linewidth=1.4,
               label="Mid", zorder=4)
    ax.axvspan(-half, half, color="grey", alpha=0.10, zorder=1)

    ax.set_title(f"{name}\nspread {snap['spread_bps']:.2f} bps  |  mid £{snap['mid_price']:.2f}",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_facecolor(BG_COLOUR)


def plot_book(snap, name, path):
    """Single-stock top-of-book depth chart."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_COLOUR)
    _draw(ax, snap, name)
    ax.set_xlabel("Distance from mid (basis points)", fontweight="bold")
    ax.set_ylabel("Depth at best price (GBP)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(snaps, path):
    """Three stocks side by side on shared scales, so spreads/depth compare."""
    max_half = max(s["spread_bps"] for s in snaps.values()) / 2.0
    max_depth = max(max(s["bid_depth_gbp"], s["ask_depth_gbp"]) for s in snaps.values())

    fig, axes = plt.subplots(1, len(snaps), figsize=(5 * len(snaps), 5), sharey=True)
    fig.patch.set_facecolor(BG_COLOUR)
    if len(snaps) == 1:
        axes = [axes]
    for ax, (stock, snap) in zip(axes, snaps.items()):
        _draw(ax, snap, config.STOCK_NAMES.get(stock, stock))
        ax.set_xlim(-max_half * 1.3, max_half * 1.3)
        ax.set_ylim(0, max_depth * 1.15)
        ax.set_xlabel("Distance from mid (bps)", fontweight="bold")
    axes[0].set_ylabel("Depth at best price (GBP)", fontweight="bold")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Top-of-book spread and depth by stock", fontsize=13,
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = cleaned_panel()
    snaps = {stock: snapshot(g) for stock, g in df.groupby(config.COLS["stock"])}
    for stock, snap in snaps.items():
        plot_book(snap, config.STOCK_NAMES.get(stock, stock),
                  config.FIGURE_DIR / f"lob_book_{stock.replace('.', '_')}.png")
    plot_comparison(snaps, config.FIGURE_DIR / "lob_spread_comparison.png")
    print(f"Saved {len(snaps)} per-stock charts + comparison to {config.FIGURE_DIR}")
    for stock, s in snaps.items():
        print(f"  {stock:7s} spread {s['spread_bps']:.2f} bps, "
              f"bid £{s['bid_depth_gbp']:,.0f}, ask £{s['ask_depth_gbp']:,.0f}")


if __name__ == "__main__":
    main()
