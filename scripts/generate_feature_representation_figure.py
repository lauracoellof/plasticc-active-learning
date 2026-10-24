"Generate feature representation figure."

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTS = (
    ROOT / "reports" / "figures" / "representacion_fotometrica_304.pdf",
    ROOT / "reports" / "figures" / "representacion_fotometrica_304.png",
)

INK = "#34495E"
INPUT = "#E8EEF3"
BLOCK = "#E6EEE8"
GP = "#E9EEF5"
META = "#F2ECD9"
OUTPUT = "#E9E5F1"
AUX = "#F3E7E2"

fig, ax = plt.subplots(figsize=(13.8, 6.7))
ax.set_xlim(0, 13.8)
ax.set_ylim(0, 6.7)
ax.axis("off")


def box(x, y, w, h, label, color, size=9.0, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.055,rounding_size=0.08",
        facecolor=color,
        edgecolor=INK,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color="#253443",
    )


def arrow(x1, y1, x2, y2, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=INK,
            linestyle=(0, (4, 3)) if dashed else "solid",
            shrinkA=2,
            shrinkB=2,
        )
    )


ax.text(
    0.2,
    6.35,
    "Construction of the 304-feature photometric representation",
    fontsize=12.2,
    fontweight="bold",
    color="#253443",
)

box(
    0.25,
    3.15,
    2.2,
    1.35,
    "Object input\ncleaned light curve\n+ photometric metadata",
    INPUT,
    size=9.2,
    weight="bold",
)

box(3.05, 5.05, 2.45, 0.88, "Statistics + metadata\n91 = 72 + 15 + 4", BLOCK, weight="bold")
box(3.05, 3.65, 2.45, 0.88, "GP1D by band\n136 = 120 + 16", GP, weight="bold")
box(3.05, 2.25, 2.45, 0.88, "GP2D time--$\\lambda$\n53 variables", GP, weight="bold")
box(6.25, 1.25, 2.35, 0.90, "GP1D + metadata\n24 transformations", META, size=8.7, weight="bold")
box(3.05, 0.55, 2.45, 0.92, "Light curves + metadata\n13 auxiliary descriptors", AUX, size=8.6)

box(
    9.10,
    2.95,
    2.15,
    1.32,
    "Join on object_id\nwith a fixed order\n91 + 136 + 53 + 24",
    BLOCK,
    size=8.8,
    weight="bold",
)
box(
    11.75,
    2.95,
    1.80,
    1.32,
    "Final vector\n$\\mathbf{x}\\in\\mathbb{R}^{304}$\nE2-GP input",
    OUTPUT,
    size=8.8,
    weight="bold",
)

arrow(2.45, 3.93, 3.05, 5.48)
arrow(2.45, 3.80, 3.05, 4.09)
arrow(2.45, 3.62, 3.05, 2.69)
arrow(5.50, 4.09, 6.25, 1.78)
arrow(2.45, 3.40, 3.05, 1.01, dashed=True)
arrow(5.50, 5.49, 9.10, 3.94)
arrow(5.50, 4.09, 9.10, 3.71)
arrow(5.50, 2.69, 9.10, 3.35)
arrow(8.60, 1.70, 9.10, 3.10)
arrow(11.25, 3.61, 11.75, 3.61)

ax.text(
    3.12,
    0.32,
    "Branch excluded from training, selection and metrics",
    fontsize=8.1,
    fontweight="bold",
    color="#8A5A4B",
)
ax.text(
    0.25,
    0.03,
    "Predictive inputs: MJD, passband, differential flux, flux error, photometric redshift and error, extinction and distance modulus. Identifiers, DDF/WFD and labels are excluded.",
    fontsize=7.8,
    color="#566573",
)

for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)
