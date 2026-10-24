"Generate two channel architecture."

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "arquitectura_dos_canales.png"

INK = "#3F5264"
COL = {
    "input": "#E7EEF4",
    "model": "#E4EEE7",
    "signal": "#F2ECD8",
    "channel": "#E9E4F1",
    "decision": "#EEF0F2",
    "oracle": "#F3E2E2",
    "output": "#E7EEF4",
    "anomaly": "#F3E7E1",
    "muted": "#788693",
}

fig, ax = plt.subplots(figsize=(15.0, 7.0))
ax.set_xlim(0, 15.0)
ax.set_ylim(0, 7.0)
ax.axis("off")


def box(x, y, w, h, text, kind, *, size=9.3, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor=COL[kind],
        edgecolor=INK,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color="#263746",
    )


def arrow(
    x1, y1, x2, y2, *, text=None, dashed=False, curve=0.0, color=INK, dx=0.0, dy=0.0, size=7.9
):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.05,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (x1 + x2) / 2 + dx,
            (y1 + y2) / 2 + dy,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )


ax.text(
    0.30,
    6.65,
    "Active classification and discovery system",
    fontsize=12.2,
    fontweight="bold",
    color="#263746",
)


box(0.30, 4.65, 2.05, 0.90, "Labelled set\n$\\mathcal{L}_t$ · known classes", "input")
box(0.30, 2.20, 2.05, 0.90, "Available pool\n$\\mathcal{U}_t$ · 304 inputs", "input")


box(
    2.95,
    4.65,
    2.30,
    0.90,
    "E2-GP classifier\nClass-balanced RF\n+ sigmoid calibration",
    "model",
    weight="bold",
)
box(2.95, 2.20, 2.30, 0.90, "Static descriptors\nIF · GP residuals\namplitude", "signal")
box(5.85, 4.65, 2.15, 0.90, "Class probabilities $p_c$\n$C$, $H$ and $1-C$", "signal")
box(
    5.85,
    2.20,
    2.15,
    0.90,
    "Anomaly reranker\n$f(1-C,\\,IF,\\,r_{GP},\\,A)$\n(initial prior: $1-C$)",
    "model",
    size=8.7,
)


box(
    8.65,
    4.80,
    2.10,
    0.95,
    "0 · Automatic removal\n$C \\geq 0.60$\nno query",
    "decision",
    weight="bold",
)
box(
    8.65,
    3.08,
    2.10,
    0.95,
    "2 · Classification\nquota $b_{clas}$ · top $H$\nremaining objects",
    "channel",
    weight="bold",
    size=8.9,
)
box(
    8.65,
    1.36,
    2.10,
    0.95,
    "1 · Discovery\nquota $b_{desc}$ · top $s_{AAD}$",
    "channel",
    weight="bold",
)

box(11.45, 2.20, 1.35, 1.05, "Simulated\noracle", "oracle", weight="bold")
box(13.20, 4.35, 1.45, 1.30, "Known label\nupdate classifier\nand reranker", "output", size=8.2)
box(13.20, 0.90, 1.45, 1.30, "99x label\ncount discovery\nupdate reranker", "anomaly", size=8.2)


arrow(2.35, 5.10, 2.95, 5.10)
arrow(2.35, 2.65, 2.95, 2.65)
arrow(2.35, 2.65, 2.95, 4.82, curve=-0.10, text="predict", dx=-0.05, dy=0.05)
arrow(5.25, 5.10, 5.85, 5.10)
arrow(5.25, 2.65, 5.85, 2.65)
arrow(6.92, 4.65, 6.92, 3.10, text="$1-C$", dx=0.28)
arrow(8.00, 4.97, 8.65, 5.25)
arrow(8.00, 4.83, 8.65, 3.55, text="$H$", dx=0.02, dy=0.06)
arrow(8.00, 2.65, 8.65, 1.88, text="$s_{AAD}$", dx=0.12, dy=-0.02)
arrow(10.75, 3.55, 11.45, 2.90)
arrow(10.75, 1.84, 11.45, 2.45)
arrow(12.80, 2.95, 13.20, 4.92, text="label", dx=-0.06, dy=0.03)
arrow(12.80, 2.42, 13.20, 1.55, text="label", dx=-0.08, dy=-0.02)

ax.text(
    0.35,
    0.48,
    "Read from left to right: estimate scores, choose queries, obtain labels and update the next round.",
    fontsize=8.5,
    color="#566573",
)
ax.text(
    0.35,
    0.20,
    "Automatic removal consumes no labels and creates no pseudo-labels. 99x labels never enter the multiclass classifier.",
    fontsize=8.5,
    color="#566573",
)

fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
print(OUT)
