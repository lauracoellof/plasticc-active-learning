"Generate validation figures."

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.models import classification
from src.models import evaluation as ev


PROCESSED = config.DIR_PROCESSED
FIGURES = config.FIGURE_DIR
BLUE = "#2F6690"
ORANGE = "#B86F3C"
TEAL = "#3D7F78"
GRAY = "#858B91"
RED = "#A94B50"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def confusion_oof() -> None:
    train = classification.features_e2gp("train")
    y_true = train["target"].to_numpy()
    cache = np.load(PROCESSED / "calibracion_e2gp_oof_150_cv3.npz")
    classes = cache["clases"]
    y_pred = classes[cache["proba_cal"].argmax(axis=1)]
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    labels = [classification.CLASS_NAMES[int(c)] for c in classes]

    fig, ax = plt.subplots(figsize=(8.3, 7.2))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.015:
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if cm[i, j] > 0.55 else "#202020",
                )
    ax.set_xticks(range(len(labels)), labels=labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Out-of-fold predictions of the canonical E2-GP classifier")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction within each true class")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_confusion_e2gp_oof.png", bbox_inches="tight")
    plt.close(fig)


def confusion_batch4() -> None:
    "Confusion batch4."
    predictions = pd.read_csv(PROCESSED / "predicciones_test_batch4.csv")
    known = predictions[predictions["true_target"] < 990].copy()
    classes = sorted(
        set(known["true_target"].astype(int).unique()) | set(known["pred"].astype(int).unique())
    )
    cm = confusion_matrix(
        known["true_target"].astype(int),
        known["pred"].astype(int),
        labels=classes,
        normalize="true",
    )
    labels = [classification.CLASS_NAMES[int(code)] for code in classes]

    fig, ax = plt.subplots(figsize=(8.3, 7.1))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.02:
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if cm[i, j] > 0.55 else "#202020",
                )
    ax.set_xticks(range(len(labels)), labels=labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("External E2-GP evaluation on batch 4")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction within each true class")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_confusion_batch4.png", bbox_inches="tight")
    plt.close(fig)


def oof_calibration() -> None:
    "Oof calibration."
    train = classification.features_e2gp("train")
    y_true = train["target"].to_numpy()
    cache = np.load(PROCESSED / "calibracion_e2gp_oof_150_cv3.npz")
    classes = cache["clases"]
    proba_uncalibrated = cache["proba_sin"]
    proba_calibrated = cache["proba_cal"]
    ece_uncalibrated, curve_uncalibrated = ev.confidence_ece(y_true, proba_uncalibrated, classes)
    ece_calibrated, curve_calibrated = ev.confidence_ece(y_true, proba_calibrated, classes)

    fig, (ax, ax_hist) = plt.subplots(
        1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for curve, label, color, marker in [
        (
            curve_uncalibrated,
            f"Uncalibrated · ECE={ece_uncalibrated:.3f}",
            BLUE,
            "o",
        ),
        (
            curve_calibrated,
            f"Platt cv=3 · ECE={ece_calibrated:.3f}",
            ORANGE,
            "s",
        ),
    ]:
        x = [point[0] for point in curve]
        y = [point[1] for point in curve]
        ax.plot(x, y, marker=marker, lw=2, label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Perfect calibration")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Empirical accuracy",
        title="E2-GP OOF reliability",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    confidence = proba_calibrated.max(axis=1)
    ax_hist.axvspan(0.00, 0.30, color="#E8EEF2", alpha=0.9)
    ax_hist.axvspan(0.30, 0.60, color="#F2EDE5", alpha=0.9)
    ax_hist.axvspan(0.60, 1.00, color="#E7EFEC", alpha=0.9)
    ax_hist.hist(
        confidence,
        bins=np.linspace(0, 1, 21),
        color=BLUE,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    for threshold, label, color in [
        (0.30, "lower threshold", RED),
        (0.60, "automatic removal", TEAL),
    ]:
        ax_hist.axvline(
            threshold,
            color=color,
            ls="--",
            lw=1.8,
            label=f"{threshold:.2f} · {label}",
        )
    ax_hist.set(
        xlabel="Calibrated OOF confidence",
        ylabel="Number of objects",
        title="Predefined operating thresholds",
    )
    ymax = ax_hist.get_ylim()[1]
    for x, label in [
        (0.15, "low-confidence tail"),
        (0.45, "intermediate region"),
        (0.80, "high confidence"),
    ]:
        ax_hist.text(x, ymax * 0.97, label, ha="center", va="top", fontsize=8.2, color="#374151")
    ax_hist.legend(frameon=False)
    ax_hist.grid(alpha=0.22)
    fig.suptitle(
        "Canonical E2-GP · 150-tree RF and three-fold sigmoid calibration",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "calibracion_E2GP_fiabilidad.png", bbox_inches="tight")
    plt.close(fig)


def batch4_domain_shift() -> None:
    "Batch4 domain shift."
    train_meta = pd.read_csv(config.METADATA_CLEAN_CSV)
    test_meta = pd.read_csv(config.TEST_METADATA_CLEAN_CSV)
    test_known = test_meta[test_meta["true_target"] < 990].copy()
    train_lc = pd.read_csv(config.TRAIN_CLEAN_CSV)
    test_lc = pd.read_csv(config.TEST_CLEAN_CSV)

    classes = sorted(train_meta["target"].unique())
    labels = [classification.CLASS_NAMES[int(code)] for code in classes]
    p_train = train_meta["target"].value_counts(normalize=True).reindex(classes)
    p_test = test_known["true_target"].value_counts(normalize=True).reindex(classes, fill_value=0)

    def object_summaries(frame: pd.DataFrame) -> pd.DataFrame:
        tmp = frame.assign(snr=np.abs(frame["flux"]) / frame["flux_err"])
        return tmp.groupby("object_id").agg(
            n_obs=("mjd", "size"),
            snr_med=("snr", "median"),
        )

    summaries = {
        "Training": object_summaries(train_lc),
        "Batch 4": object_summaries(test_lc),
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.4, 5.5),
        gridspec_kw={"width_ratios": [1.45, 1.0, 1.0], "wspace": 0.34},
    )
    ax = axes[0]
    y = np.arange(len(classes))
    for yi, left, right in zip(y, p_train, p_test):
        ax.plot([left, right], [yi, yi], color="#C7CDD2", lw=1.2, zorder=1)
    ax.scatter(
        p_train,
        y,
        marker="s",
        s=34,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=1.2,
        label="Training",
        zorder=3,
    )
    ax.scatter(
        p_test,
        y,
        marker="o",
        s=34,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        label="Batch 4",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Prevalence among known classes")
    ax.set_title("(a) Class composition", loc="left")
    ax.grid(axis="x", which="major", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")

    for ax, column, xlabel, panel in [
        (axes[1], "n_obs", "Retained observations per object", "(b) Coverage"),
        (axes[2], "snr_med", "Median of $|f|/\\sigma_f$ by object", "(c) SNR"),
    ]:
        for name, color in [("Training", GRAY), ("Batch 4", BLUE)]:
            values = np.sort(summaries[name][column].to_numpy(float))
            ecdf = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, ecdf, color=color, lw=2, label=name)
            median = float(np.median(values))
            ax.axvline(median, color=color, lw=1, ls=(0, (3, 3)), alpha=0.85)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(panel, loc="left")
        ax.grid(alpha=0.22)
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("Observable domain shift between training and batch 4", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_desajuste_dominio_batch4.png", bbox_inches="tight")
    plt.close(fig)


def anomaly_descriptors() -> None:
    consistency = pd.read_csv(PROCESSED / "anomalias_consistencia_batches.csv")
    consistency = consistency[consistency["batch"].astype(str) != "media"].copy()
    columns = ["IsolationForest", "residuos_GP", "novedad_1-conf", "cambios_mag"]
    names = ["Isolation Forest", "GP residuals", "1 − confidence", "Amplitude"]
    subtypes = pd.read_csv(PROCESSED / "anomalias_complementariedad_pool.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.35), gridspec_kw={"width_ratios": [0.95, 1.3]})
    ax = axes[0]
    x = np.arange(len(columns))
    batch_colors = [BLUE, "#6B7280", TEAL, "#7D5A50"]
    batch_markers = ["o", "s", "^", "v"]
    for (_, row), color, marker in zip(consistency.iterrows(), batch_colors, batch_markers):
        label = f"batch {row['batch']}" + (" (DDF)" if str(row["batch"]) == "1" else "")
        ax.plot(
            x,
            row[columns].astype(float),
            marker=marker,
            ms=5.5,
            lw=1.5,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label=label,
            zorder=3,
        )
    means = consistency[columns].astype(float).mean(axis=0).to_numpy()
    ax.plot(x, means, color="#111111", marker="D", ms=5.5, lw=2.1, label="mean", zorder=4)
    ax.axhline(0.5, color=GRAY, ls="--", lw=1)
    ax.set_xticks(x, names, rotation=24, ha="right")
    ax.set_ylim(0.34, 0.76)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Consistency across batches")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, frameon=False, loc="upper left")

    ax = axes[1]
    matrix = subtypes[columns].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("tfm", ["#F4F1EA", "#7FB3C8", BLUE])
    heat = ax.imshow(matrix, cmap=cmap, vmin=0.25, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(columns)), names, rotation=24, ha="right")
    ax.set_yticks(
        range(len(subtypes)),
        [f"{int(r.anomalia)}  (n={int(r.n)})" for _, r in subtypes.iterrows()],
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.67 else "#202020",
                fontsize=9,
            )
    ax.set_title("Complementarity by 99x subtype")
    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ROC-AUC")
    fig.suptitle("Information from anomaly descriptors", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_descriptores_anomalia.png", bbox_inches="tight")
    plt.close(fig)


def query_strategies() -> None:
    data = pd.read_csv(PROCESSED / "estrategias_consulta_batch4_5seeds.csv")
    final = data.sort_values("iter").groupby(["estrategia", "semilla"]).tail(1)
    order = ["entropy", "least_confidence", "margin", "random"]
    labels = ["Entropy", "1 − confidence", "Margen", "Random"]
    summary = (
        final.groupby("estrategia")
        .agg(
            recall=("recall_anom", "mean"),
            recall_sd=("recall_anom", "std"),
            mf1=("MF1", "mean"),
            mf1_sd=("MF1", "std"),
            n=("semilla", "nunique"),
        )
        .loc[order]
    )
    summary["recall_se"] = summary["recall_sd"] / np.sqrt(summary["n"])
    summary["mf1_se"] = summary["mf1_sd"] / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(6.4, 4.25))
    colors = [TEAL, BLUE, RED, GRAY]
    x = np.arange(len(order))
    values = summary["mf1"].to_numpy()
    errors = summary["mf1_se"].to_numpy()
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=3.5,
        color=colors,
        width=0.68,
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, value, error_value in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error_value + max(values) * 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", alpha=0.22)
    lower = max(0.0, float(values.min() - 0.12 * (values.max() - values.min())))
    ax.set_ylim(lower, float(np.max(values + errors) * 1.12))
    fig.suptitle("Batch-4 classification by query strategy · five seeds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_estrategias_consulta.png", bbox_inches="tight")
    plt.close(fig)


def batch4_systems() -> None:
    two = pd.read_csv(PROCESSED / "dos_canales_batch4_10seeds.csv")
    triage = pd.read_csv(PROCESSED / "triaje_batch4_10seeds.csv")
    data = pd.concat(
        [
            two[np.isclose(two["frac_desc"], 0.6)].assign(sistema="Two channels"),
            triage[triage["modo"] == "triage"].assign(sistema="Triage"),
            triage[triage["modo"] == "random"].assign(sistema="Random"),
        ],
        ignore_index=True,
    )
    summary = (
        data.groupby(["sistema", "n_consultas"])
        .agg(
            recall=("recall_anom", "mean"),
            recall_sd=("recall_anom", "std"),
            mf1=("MF1", "mean"),
            mf1_sd=("MF1", "std"),
            n=("semilla", "nunique"),
        )
        .reset_index()
    )
    summary["recall_se"] = summary["recall_sd"] / np.sqrt(summary["n"])
    summary["mf1_se"] = summary["mf1_sd"] / np.sqrt(summary["n"])

    palette = {"Two channels": BLUE, "Triage": ORANGE, "Random": GRAY}
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.25))
    for ax, metric, error, ylabel, title in [
        (axes[0], "recall", "recall_se", "Recall 99x", "Discovery"),
        (axes[1], "mf1", "mf1_se", "Macro-F1", "Classification"),
    ]:
        for system in ["Two channels", "Triage", "Random"]:
            subset = summary[summary["sistema"] == system].sort_values("n_consultas")
            x = subset["n_consultas"].to_numpy()
            y = subset[metric].to_numpy()
            e = subset[error].to_numpy()
            ax.plot(x, y, marker="o", ms=3.7, lw=2, color=palette[system], label=system)
            ax.fill_between(x, y - e, y + e, color=palette[system], alpha=0.14)
        ax.set_xlabel("Cumulative queries")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        if metric == "recall":
            ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("Batch-4 efficiency per query · ten seeds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_sistemas_batch4.png", bbox_inches="tight")
    plt.close(fig)


def batch_generalization() -> None:
    data = pd.read_csv(PROCESSED / "generalizacion_por_batch.csv")
    n = data["semilla"].nunique()
    summary = (
        data.groupby(["batch", "sistema"])
        .agg(
            mf1=("MF1", "mean"),
            mf1_sd=("MF1", "std"),
            recall=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
        .reset_index()
    )
    summary["mf1_se"] = summary["mf1_sd"] / np.sqrt(n)
    summary["recall_se"] = summary["recall_sd"] / np.sqrt(n)

    order = ["two channels", "triage", "random"]
    labels = {"two channels": "Two channels", "triage": "Triage", "random": "Random"}
    palette = {"two channels": BLUE, "triage": ORANGE, "random": GRAY}
    batches = sorted(summary["batch"].unique())
    x = np.arange(len(batches))
    width = 0.24
    offsets = {"two channels": -width, "triage": 0.0, "random": width}
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.25))
    for ax, metric, error, ylabel, title in [
        (axes[0], "mf1", "mf1_se", "Macro-F1", "Classification"),
        (axes[1], "recall", "recall_se", "Recall 99x", "Discovery"),
    ]:
        for system in order:
            subset = summary[summary["sistema"] == system].set_index("batch").loc[batches]
            ax.bar(
                x + offsets[system],
                subset[metric],
                width=width,
                yerr=subset[error],
                capsize=3.0,
                color=palette[system],
                edgecolor="white",
                linewidth=0.55,
                label=labels[system],
            )
        ax.set_xticks(x, [f"batch {b}" for b in batches])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        if metric == "recall":
            ax.set_ylim(0, float(summary[metric].max() * 1.18))
        else:
            ax.set_ylim(0.58, float(summary[metric].max() * 1.08))
        if metric == "recall":
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Comparison across independent batches · α = 0.6 · ten seeds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_generalizacion_batches.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    oof_calibration()
    confusion_oof()
    confusion_batch4()
    batch4_domain_shift()
    anomaly_descriptors()
    query_strategies()
    batch4_systems()
    batch_generalization()
    for name in [
        "calibracion_E2GP_fiabilidad.png",
        "06_confusion_e2gp_oof.png",
        "06_confusion_batch4.png",
        "06_desajuste_dominio_batch4.png",
        "06_descriptores_anomalia.png",
        "06_estrategias_consulta.png",
        "06_sistemas_batch4.png",
        "06_generalizacion_batches.png",
    ]:
        print(FIGURES / name)
