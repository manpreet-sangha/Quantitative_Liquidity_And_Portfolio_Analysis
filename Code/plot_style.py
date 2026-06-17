"""Shared, high-quality matplotlib styling for the Part 1 report figures.

Goals: legible text when figures are embedded small in the report, strong contrast,
and a colourblind-safe palette. ``save_fig`` writes a vector PDF (used in the report,
sharp at any zoom) and a high-resolution PNG (used by the Streamlit app and as a
fallback). Import this *after* ``matplotlib.use("Agg")`` so the headless backend is
kept.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# Okabe-Ito colourblind-safe, high-contrast palette. With alphabetical groupby the
# three default stocks map to EXPN=blue, KGF=vermillion, ULVR=green.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def apply_style():
    """Apply the shared report plotting style (fonts, palette, grid, white background)."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,            # crisp high-resolution PNG (report uses vector PDF)
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,            # embed real (selectable, sharp) fonts in the PDF
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#666666",
    })


def save_fig(fig, figure_dir, stem):
    """Save ``fig`` as both vector PDF (report) and high-res PNG (app). Returns PNG path."""
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    return png_path
