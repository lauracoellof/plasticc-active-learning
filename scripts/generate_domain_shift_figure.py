"Generate domain shift figure."

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "figures" / "01_train_vs_test.png"

NAMES = {
    6: "$\\mu$Lens",
    15: "TDE",
    16: "EB",
    42: "SNII",
    52: "SNIax",
    53: "Mira",
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    92: "RRL",
    95: "SLSN-I",
}

train = pd.read_csv(RAW / "training_set_metadata.csv", usecols=["target"])["target"]
test = pd.read_csv(RAW / "plasticc_test_metadata.csv.gz", usecols=["true_target"])["true_target"]

classes = sorted(train.unique())
p_train = train.value_counts(normalize=True).reindex(classes).astype(float)
p_test = test.value_counts(normalize=True).reindex(classes).astype(float)
log_ratio = np.log2(p_test.to_numpy() / p_train.to_numpy())
labels = [NAMES[int(c)] for c in classes]
novel = float(test.isin([991, 992, 993, 994]).mean())

plt.rcParams.update(
    {
        "font.size": 10.3,
        "axes.titlesize": 11.2,
        "axes.labelsize": 10.3,
        "legend.fontsize": 9.3,
        "axes.edgecolor": "#4B5563",
    }
)


fig, (ax0, ax1) = plt.subplots(
    1,
    2,
    figsize=(12.4, 6.1),
    gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.30},
)
y = np.arange(len(classes))
navy = "#1F4E79"
slate = "#66788A"
connector = "#B6BEC6"

for yi, ptr, pte in zip(y, p_train, p_test):
    ax0.plot([ptr, pte], [yi, yi], color=connector, linewidth=1.25, zorder=1)
ax0.scatter(
    p_train,
    y,
    s=42,
    marker="s",
    facecolors="white",
    edgecolors=slate,
    linewidths=1.4,
    label="Training",
    zorder=3,
)
ax0.scatter(
    p_test,
    y,
    s=42,
    marker="o",
    color=navy,
    edgecolors="white",
    linewidths=0.5,
    label="Test",
    zorder=4,
)
ax0.set_xscale("log")
ax0.set_yticks(y, labels)
ax0.invert_yaxis()
ax0.set_xlabel("Catalogue prevalence (logarithmic scale)")
ax0.set_title("(a) Absolute prevalence", loc="left", fontweight="bold")
ax0.grid(axis="x", which="major", color="#D9DEE3", linewidth=0.8)
ax0.grid(axis="x", which="minor", color="#EEF1F3", linewidth=0.5)
ax0.legend(frameon=False, loc="lower right")

for yi, value in zip(y, log_ratio):
    ax1.plot([0, value], [yi, yi], color=connector, linewidth=1.25, zorder=1)
ax1.scatter(
    log_ratio, y, s=42, marker="o", color=navy, edgecolors="white", linewidths=0.5, zorder=3
)
ax1.axvline(0, color="#374151", linewidth=1.2)
for value in (-1, 1):
    ax1.axvline(value, color="#9CA3AF", linewidth=0.8, linestyle=(0, (3, 3)))
ax1.set_yticks(y, labels)
ax1.invert_yaxis()
ax1.set_xlabel("Relative change: $\\log_2[p(\\mathrm{test})/p(\\mathrm{train})]$")
ax1.set_title("(b) Domain shift", loc="left", fontweight="bold")
ax1.grid(axis="x", color="#E2E6EA", linewidth=0.7)

fig.text(
    0.5,
    0.018,
    f"99x classes without a training counterpart: {novel:.2%} of the test catalogue",
    ha="center",
    color="#374151",
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "#F3F4F6",
        "edgecolor": "#CBD1D7",
        "linewidth": 0.8,
    },
)
fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.13)
fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
print(OUT)
