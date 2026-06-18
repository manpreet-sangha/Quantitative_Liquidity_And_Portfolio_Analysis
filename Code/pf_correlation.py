"""Part 2 - rolling correlation heatmap of country returns.

At each month we take the most recent ``window`` monthly returns (the same 60-month
window the optimiser uses to build its covariance matrix) and compute the pairwise
correlation matrix across the 34 countries. The Streamlit app drives this with a
month slider so the structure can be inspected as it tightens in the 2008 and 2020
crises (the grid reddens, diversification evaporates) and relaxes afterwards. A GIF
of the full sweep is also available on demand.

Country order is fixed (alphabetical) across every view so each cell always refers to
the same pair and its colour is comparable through time.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pf_config
import pf_data

OUT = pf_config.OUTPUT_DIR
WINDOW = pf_config.WINDOW       # 60 months, matches the optimiser's covariance window
TARGET_FRAMES = 90              # cap on the number of GIF frames
FPS = 6
CMAP = "YlOrRd"


def country_returns(returns):
    """Country columns in a fixed (alphabetical) order, and their returns frame."""
    countries = sorted(pf_data.country_columns(returns))
    return countries, returns[countries]


def window_end_dates(R, window=WINDOW):
    """Every month for which a full `window`-month look-back exists."""
    return list(R.index[window - 1:])


def correlation_at(R, end_date, window=WINDOW):
    """Correlation matrix over the `window` months ending at end_date (inclusive)."""
    pos = R.index.get_loc(end_date)
    return R.iloc[pos - window + 1:pos + 1].corr()


def _draw(ax, corr, countries, date, window):
    """Draw one correlation heatmap on ax and return the image artist."""
    n = len(countries)
    im = ax.imshow(corr.values, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(countries, rotation=90, fontsize=5)
    ax.set_yticklabels(countries, fontsize=5)
    ax.set_title(f"{window}-month rolling correlation of country returns "
                 f"(window ending {date:%Y-%m})")
    return im


def heatmap_figure(returns, end_date, window=WINDOW):
    """A single static heatmap figure for the window ending end_date (for st.pyplot)."""
    countries, R = country_returns(returns)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = _draw(ax, correlation_at(R, end_date, window), countries, end_date, window)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise correlation")
    fig.tight_layout()
    return fig


def animate(returns, path, window=WINDOW, target_frames=TARGET_FRAMES):
    """Build and save the rolling-correlation heatmap GIF (dpi pinned for the cloud)."""
    countries, R = country_returns(returns)
    dates = window_end_dates(R, window)
    step = max(1, len(dates) // target_frames)
    dates = dates[::step]
    if dates[-1] != R.index[-1]:
        dates.append(R.index[-1])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = _draw(ax, correlation_at(R, dates[0], window), countries, dates[0], window)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise correlation")
    fig.tight_layout()

    def update(k):
        d = dates[k]
        im.set_data(correlation_at(R, d, window).values)
        ax.set_title(f"{window}-month rolling correlation of country returns "
                     f"(window ending {d:%Y-%m})")
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(dates),
                                   interval=1000 // FPS, blit=False)
    anim.save(str(path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    return path


def main():
    returns = pf_data.load_returns()
    out = OUT / "corr_rolling.gif"
    animate(returns, out)
    print(f"Saved rolling correlation heatmap to {out}")


if __name__ == "__main__":
    main()
