"""Part 1: animated intraday GIFs of spread and depth (for the Streamlit app).

For the selected stocks, each stock's average spread and depth by time of day is
revealed progressively along the time axis, one frame every 0.25 seconds. Two GIFs
are produced (spread and depth). Everything is config-driven, so the GIFs are built
for whatever three stocks config.PART1_STOCKS holds (no hard-coding).
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
import intraday_patterns

OUT = config.OUTPUT_DIR / "gifs"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 4              # 0.25 seconds per frame
TARGET_FRAMES = 60   # number of reveal steps along the trading day


def _intraday_panel():
    """Average spread/depth/volume by stock and minute-of-day for the chosen stocks."""
    df = data_loader.load_part1()
    df = liquidity_measures.add_measures(df)
    df, _ = cleaning.apply_filters(df)
    return intraday_patterns.by_minute_of_day(df)


def _animate(agg, col, ylabel, title, path):
    stock_col = config.COLS["stock"]
    stocks = list(agg[stock_col].unique())
    minutes = np.sort(agg["minute_of_day"].unique())
    hours = minutes / 60.0
    series = {
        s: agg[agg[stock_col] == s].set_index("minute_of_day")[col].reindex(minutes).to_numpy()
        for s in stocks
    }

    step = max(1, len(minutes) // TARGET_FRAMES)
    frames = list(range(step, len(minutes) + 1, step))
    if frames[-1] != len(minutes):
        frames.append(len(minutes))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colours = plt.cm.tab10.colors
    lines = {}
    for i, s in enumerate(stocks):
        (lines[s],) = ax.plot([], [], color=colours[i % 10],
                              label=config.STOCK_NAMES.get(s, s))
    ymax = max(np.nanmax(series[s]) for s in stocks) * 1.1
    ax.set_xlim(hours.min(), hours.max())
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    def update(k):
        for s in stocks:
            lines[s].set_data(hours[:k], series[s][:k])
        return list(lines.values())

    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 // FPS, blit=True)
    # Pin the frame resolution. Animations must NOT inherit the report figures' high
    # savefig.dpi (set globally by plot_style.apply_style); the Pillow writer holds
    # every frame in memory, so a high dpi can exhaust a small Streamlit Cloud instance.
    anim.save(str(path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    return path


def main():
    agg = _intraday_panel()
    _animate(agg, "spread_bps", "Mean spread (bps)",
             "Intraday spread by time of day", OUT / "intraday_spread.gif")
    _animate(agg, "depth_gbp", "Mean depth (GBP)",
             "Intraday depth by time of day", OUT / "intraday_depth.gif")
    _animate(agg, "volume", "Mean volume (shares/min)",
             "Intraday volume by time of day", OUT / "intraday_volume.gif")
    print(f"Saved intraday spread, depth & volume GIFs for "
          f"{', '.join(config.PART1_STOCKS)} to {OUT}")


if __name__ == "__main__":
    main()
