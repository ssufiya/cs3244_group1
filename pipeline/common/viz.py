"""
Shared matplotlib styling for every figure in reports/eda/figures.

Palette is the validated default categorical palette (slots assigned in fixed
order, never cycled). Slots 1-3 clear the all-pairs colour-vision gates; the
grouped-bar figures use the adjacent pairlist and additionally carry direct
value labels, which satisfies the relief rule for the low-contrast slots.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Categorical slots, fixed order — never cycled, never reordered per chart.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]

# The project's one recurring binary encoding. Fixed so that "sarcastic" is
# the same colour in every figure of the report.
C_NOT = SERIES[0]      # blue   — label 0, not sarcastic
C_SARC = SERIES[1]     # orange — label 1, sarcastic
LABEL_NAMES = {0: "Not sarcastic", 1: "Sarcastic"}
LABEL_COLORS = {0: C_NOT, 1: C_SARC}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e4e3df"

# Sequential blue ramp, light -> dark (for magnitude encodings).
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 26,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def despine(ax, left: bool = False, bottom: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not left)
    ax.spines["bottom"].set_visible(not bottom)


def subtitle(ax, text: str) -> None:
    """A recessive one-line note under the title — says what to read."""
    ax.text(0.0, 1.012, text, transform=ax.transAxes, fontsize=9,
            color=INK_MUTED, ha="left", va="bottom")


def bar_labels(ax, bars, fmt="{:.0f}", pad=2, color=INK_2, fontsize=8.5) -> None:
    """Direct labels on bars — required for the low-contrast palette slots."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, pad),
                    ha="center", va="bottom", fontsize=fontsize, color=color)


def save(fig, path, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig] {path.name}{('  — ' + title) if title else ''}")
