"""Feature-block ablations evaluated on a shared object intersection."""

import numpy as np
import pandas as pd

import config
from src.features import statistics as est
from src.models.classification import CLASS_NAMES


META_PODADA = ["distmod", "hostgal_photoz", "hostgal_photoz_err", "mwebv"]
STATS_PODADA = ["baseline_days", "spectral_centroid", "red_fraction"]

VARIANTS = ("baseline", "gp1", "gp1gp2", "podada")


def _common_table():
    """Join all feature blocks on the same object intersection before comparing representations."""
    df = pd.read_csv(config.FEATURES_CSV)
    gp1 = pd.read_csv(config.FEATURES_GP_CSV)
    gp2 = pd.read_csv(config.FEATURES_GP2D_CSV)
    d = est.add_derived_features(df.merge(gp1, on="object_id").merge(gp2, on="object_id"))
    cols_stats = [c for c in df.columns if c not in ("object_id", "target")]
    cols_gp1 = [c for c in gp1.columns if c != "object_id"]
    cols_gp2 = [c for c in gp2.columns if c != "object_id"]
    cols_deriv = [c for c in d.columns if c.endswith("_rest") or "peak_absmag" in c]
    return d, cols_stats, cols_gp1, cols_gp2, cols_deriv


def pruned_columns(d):
    """Return the fixed reduced feature subset, preserving its historical order."""
    cols = (
        [c for c in d.columns if c.startswith("gp2d_")]
        + [c for c in d.columns if "peak_absmag" in c]
        + [c for c in META_PODADA if c in d.columns]
        + [c for c in STATS_PODADA if c in d.columns]
        + [c for c in d.columns if c.startswith("min_") or c.startswith("max_")]
    )
    return list(dict.fromkeys(cols))


def variant_features(variant_name):
    """Return X, y and ordered feature names for a supported representation variant."""
    d, cols_stats, cols_gp1, cols_gp2, cols_deriv = _common_table()
    if variant_name == "baseline":
        cols = cols_stats
    elif variant_name == "gp1":
        cols = cols_stats + cols_gp1 + cols_deriv
    elif variant_name == "gp1gp2":
        cols = cols_stats + cols_gp1 + cols_gp2 + cols_deriv
    elif variant_name == "podada":
        cols = pruned_columns(d)
    else:
        raise ValueError("Unknown representation variant: %r" % variant_name)
    cols = list(dict.fromkeys(cols))
    return d[cols].values, d["target"].values, cols


def named_confusion_matrix(y_true, y_pred, plot_title, ax=None, save=None):
    """Plot a row-normalized confusion matrix with the known-class astronomy labels."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    class_codes = sorted(np.unique(y_true))
    display_names = [CLASS_NAMES.get(c, str(c)) for c in class_codes]
    cm = confusion_matrix(y_true, y_pred, labels=class_codes, normalize="true")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    else:
        fig = ax.figure
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_codes)))
    ax.set_xticklabels(display_names, rotation=90)
    ax.set_yticks(range(len(class_codes)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(plot_title)
    for i in range(len(class_codes)):
        for j in range(len(class_codes)):
            if cm[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    "%.2f" % cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150)
    return cm
