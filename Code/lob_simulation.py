"""Part 1: limit order book simulation calibrated to each stock's real data.

Runs the author's LOB simulation engine
(SMM921_Coursework/limit_order_book_simulation) but calibrates it to the REAL
input-file data for each selected stock:
  - initial price  = the stock's mean midquote (pence);
  - level spacing   = half the stock's mean spread, so the simulated best bid/ask
                      reproduce the stock's real spread;
  - order sizes     = the stock's real average top-of-book size (shares).
The order flow itself is simulated; only its scale (price, spread, depth) is set
from the data. Produces a market-depth GIF and a static snapshot per stock in
config.PART1_STOCKS, so it adapts automatically if the three stocks are re-chosen.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: we save files, never open a window

import config
import data_loader
import liquidity_measures
import cleaning

# Make the cloned simulation package importable.
REPO = Path(__file__).resolve().parents[1] / "limit_order_book_simulation"
sys.path.insert(0, str(REPO))
from src.order_book import LimitOrderBook                      # noqa: E402
from src.synthetic_data import SyntheticOrderGenerator, GeneratorConfig  # noqa: E402
from src.visualiser import LOBVisualiser, VisualiserConfig     # noqa: E402

# The visualiser forces an interactive (Tk) backend at import, which crashes under
# Streamlit's worker threads. Force the headless Agg backend back on.
import matplotlib.pyplot as plt                                # noqa: E402
plt.switch_backend("Agg")

OUT = config.OUTPUT_DIR / "lob_sim"
OUT.mkdir(parents=True, exist_ok=True)

ORDERS_PER_FRAME = 8   # process several orders per frame so the book visibly moves


def real_stats(g):
    """Mean mid (pence), mean spread (pence) and mean top-of-book size (shares)."""
    c = config.COLS
    return {
        "mid": g["mid"].mean(),
        "spread": (g[c["ask"]] - g[c["bid"]]).mean(),
        "size": 0.5 * (g[c["bid_size"]].mean() + g[c["ask_size"]].mean()),
    }


def calibrated_config(stats):
    """Build a GeneratorConfig scaled to one stock's real price/spread/size."""
    half = max(stats["spread"] / 2.0, 1e-4)        # level spacing = half spread
    avg = max(stats["size"], 1)
    # Order flow is biased to refilling near the touch and uses light market
    # orders, so the simulated spread stays close to the real (calibrated) spread.
    return GeneratorConfig(
        initial_price=round(stats["mid"], 4),
        tick_size=round(half, 4),
        n_initial_levels=10,
        initial_vol_min=max(1, int(0.5 * avg)),
        initial_vol_max=max(2, int(1.5 * avg)),
        prob_limit=0.64, prob_market=0.12, prob_cancel=0.24,
        limit_spread_ticks=5,
        limit_vol_min=max(1, int(0.3 * avg)),
        limit_vol_max=max(2, int(1.2 * avg)),
        market_vol_min=max(1, int(0.2 * avg)),
        market_vol_max=max(2, int(0.5 * avg)),
        seed=config.RANDOM_SEED,
    )


def _build(stock, g, name):
    """Return (visualiser, next_fn, cfg) for one calibrated stock simulation."""
    cfg = calibrated_config(real_stats(g))
    book = LimitOrderBook()
    gen = SyntheticOrderGenerator(cfg, book)
    gen.seed_book()
    stream = gen.stream()
    tick = [0]

    def nxt():
        snap = book.snapshot()
        for _ in range(ORDERS_PER_FRAME):
            tick[0] += 1
            snap = book.process_order(next(stream), timestamp=tick[0])
        return snap

    title = (f"{name}: LOB calibrated to real data "
             f"(mid {cfg.initial_price:.0f}p, spread {2 * cfg.tick_size:.2f}p)")
    vis = LOBVisualiser(VisualiserConfig(fig_width=14, fig_height=7, title=title))
    return vis, nxt, cfg


def run_stock(stock, g, name, seconds=8.0, fps=12, warmup=400):
    """Save a static snapshot PNG and an animated GIF for one stock."""
    base = stock.replace(".", "_")

    # Static snapshot: warm the book up, then draw one frame.
    vis, nxt, cfg = _build(stock, g, name)
    for _ in range(warmup // ORDERS_PER_FRAME):
        snap = nxt()
    vis._setup_figure()
    vis._next = lambda: snap
    vis._update(0)
    png = OUT / f"lob_sim_{base}.png"
    vis._fig.savefig(str(png), dpi=130, facecolor=vis.cfg.bg_colour, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(vis._fig)

    # Animated GIF: fresh calibrated run.
    vis, nxt, cfg = _build(stock, g, name)
    gif = OUT / f"lob_sim_{base}.gif"
    vis.save_gif(nxt, filepath=str(gif), duration_s=seconds, fps=fps)
    return cfg, png, gif


def main():
    df = data_loader.load_part1()
    df = liquidity_measures.add_measures(df)
    df, _ = cleaning.apply_filters(df)

    for stock, g in df.groupby(config.COLS["stock"]):
        name = config.STOCK_NAMES.get(stock, stock)
        cfg, png, gif = run_stock(stock, g, name)
        print(f"{stock:7s} mid {cfg.initial_price:.0f}p  spread {2 * cfg.tick_size:.2f}p  "
              f"-> {png.name}, {gif.name}")


if __name__ == "__main__":
    main()
