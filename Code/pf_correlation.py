"""Part 2 - animated rolling correlation heatmap of country returns.

At each step in time we take the most recent ``window`` monthly returns (the same
60-month window the optimiser uses to build its covariance matrix) and compute the
pairwise correlation matrix across the 34 countries. Animating these matrices as the
window slides forward shows how the cross-country correlation structure changes over
the sample: it tightens (the grid reddens) in the 2008 and 2020 crises as everything
starts to move together and diversification evaporates, then relaxes afterwards.

The output is a GIF for the Streamlit app. As with the Part 1 animations the frame
resolution is pinned (dpi=100) so the Pillow writer, which holds every frame in memory,
does not exhaust a small Streamlit Cloud instance.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pf_config
import pf_data

OUT = pf_config.OUTPUT_DIR
WINDOW = pf_config.WINDOW       # 60 months, matches the optimiser's covariance window
TARGET_FRAMES = 90              # cap on the number of animation frames
FPS = 6


def rolling_frames(returns, window=WINDOW, target_frames=TARGET_FRAMES):
    """Return (countries, frames) where each frame is (end_date, correlation matrix).

    Countries are held in a single fixed order across every frame so that each cell of
    the grid always refers to the same pair and its colour change over time is meaningful.
    """
    countries = sorted(pf_data.country_columns(returns))
    R = returns[countries]
    ends = list(range(window, len(R) + 1))          # exclusive end index of each window
    step = max(1, len(ends) // target_frames)
    ends = ends[::step]
    if ends[-1] != len(R):
        ends.append(len(R))
    frames = [(R.index[e - 1], R.iloc[e - window:e].corr()) for e in ends]
    return countries, frames


def animate(returns, path, window=WINDOW):
    """Build and save the rolling-correlation heatmap GIF (dpi pinned for the cloud)."""
    countries, frames = rolling_frames(returns, window)
    n = len(countries)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(frames[0][1].values, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise correlation")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(countries, rotation=90, fontsize=5)
    ax.set_yticklabels(countries, fontsize=5)
    title = ax.set_title("")
    fig.tight_layout()

    def update(k):
        date, corr = frames[k]
        im.set_data(corr.values)
        title.set_text(f"{window}-month rolling correlation of country returns "
                       f"(window ending {date:%Y-%m})")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 // FPS, blit=False)
    anim.save(str(path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    return path


def main():
    returns = pf_data.load_returns()
    out = OUT / "corr_rolling.gif"
    animate(returns, out)
    print(f"Saved rolling correlation heatmap ({TARGET_FRAMES} frames max) to {out}")


if __name__ == "__main__":
    main()
