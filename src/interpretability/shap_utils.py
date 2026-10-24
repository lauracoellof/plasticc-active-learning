"""Explain and plot the forests underlying the calibrated classifier.

TreeExplainer values refer to pre-calibration forest outputs. They do not
decompose the final calibrated probabilities or the operational novelty score."""

import numpy as np


OKABE_ITO = {
    "azul": "#0072B2",
    "naranja": "#E69F00",
    "verde": "#009E73",
    "rojo": "#D55E00",
    "gris": "#999999",
    "morado": "#CC79A7",
    "celeste": "#56B4E9",
    "amarillo": "#F0E442",
}


FAMILY_COLORS = {
    "Statistics": OKABE_ITO["gris"],
    "GP 1D": OKABE_ITO["azul"],
    "GP 2D": OKABE_ITO["verde"],
    "Derived physical features": OKABE_ITO["naranja"],
}


def feature_family(col):
    """Assign a feature to its plotting family using the fixed feature-name prefixes."""
    if col.startswith("gp2d_"):
        return "GP 2D"
    if (
        col in ("distmod", "hostgal_photoz", "hostgal_photoz_err")
        or col.startswith("gp_peak_absmag")
        or col.endswith("_rest")
    ):
        return "Derived physical features"
    if col.startswith("gp_"):
        return "GP 1D"
    return "Statistics"


def thesis_plot_style():
    """Apply the common matplotlib style used for the interpretability figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
        }
    )


def underlying_forests(estimator_model):
    """Return the fitted forest estimators underlying an ordinary or calibrated classifier."""
    from sklearn.ensemble import RandomForestClassifier

    if isinstance(estimator_model, RandomForestClassifier):
        return [estimator_model]
    cc = getattr(estimator_model, "calibrated_classifiers_", None)
    if cc is None:
        raise TypeError(
            "Expected a fitted CalibratedClassifierCV or RandomForestClassifier; received %s"
            % type(estimator_model).__name__
        )
    bases = []
    for c in cc:
        est = getattr(c, "estimator", None)
        if est is None:
            est = getattr(c, "base_estimator", None)
        if est is None:
            raise AttributeError("The base estimator of a calibration fold could not be found.")
        bases.append(est)
    return bases


def explain_model(
    estimator_model,
    X,
    feature_names_list=None,
    class_names_list=None,
    max_samples=None,
    random_state=42,
):
    "Explain the underlying forests before calibration.\n\nAverage TreeExplainer attributions across the calibration folds. These values\nexplain the mean output of the uncalibrated forests, not the final calibrated\nprobabilities. Return the explanation and the selected sample indices."
    import shap

    X = np.asarray(X, dtype=float)
    idx = np.arange(X.shape[0])
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(X.shape[0], size=max_samples, replace=False))
        X = X[idx]

    valores, bases = [], []
    for rf in underlying_forests(estimator_model):
        expl = shap.TreeExplainer(rf)
        sv = np.asarray(expl.shap_values(X))
        if sv.ndim == 2:
            sv = np.stack([-sv, sv], axis=-1)
        ev = np.atleast_1d(np.asarray(expl.expected_value, dtype=float))
        valores.append(sv)
        bases.append(ev)

    values = np.mean(valores, axis=0)
    base_values = np.mean(bases, axis=0)
    base_values = np.tile(base_values, (values.shape[0], 1))

    e = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X,
        feature_names=list(feature_names_list) if feature_names_list is not None else None,
    )
    e.class_names = list(class_names_list) if class_names_list is not None else None
    return e, idx


def _feature_names(expl, n):
    return (
        list(expl.feature_names) if expl.feature_names is not None else [f"f{i}" for i in range(n)]
    )


def global_importance(expl, top=20, color_by_family=False, ax=None, plot_title=None, save=None):
    """Plot mean absolute pre-calibration SHAP values, optionally coloured by feature family."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vals = np.abs(expl.values).mean(axis=(0, 2))
    display_names = _feature_names(expl, len(vals))
    imp = (
        pd.DataFrame({"feature": display_names, "importancia": vals})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    d = imp.head(top).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 0.34 * len(d) + 1))
    if color_by_family:
        cols = [FAMILY_COLORS[feature_family(f)] for f in d["feature"]]
        ax.barh(d["feature"], d["importancia"], color=cols, edgecolor="white", linewidth=0.4)
        presentes = [
            k
            for k in ["Derived physical features", "GP 2D", "GP 1D", "Statistics"]
            if any(feature_family(f) == k for f in d["feature"])
        ]
        handles = [
            Line2D([0], [0], marker="s", ls="", ms=9, mfc=FAMILY_COLORS[k], mec="none", label=k)
            for k in presentes
        ]
        ax.legend(handles=handles, title="Feature family", loc="lower right", frameon=False)
    else:
        ax.barh(d["feature"], d["importancia"], color=OKABE_ITO["azul"])
    ax.set_xlabel("Mean absolute SHAP value (averaged across classes)")
    if plot_title:
        ax.set_title(plot_title)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.01)
    plt.tight_layout()
    if save is not None:
        ax.figure.savefig(save, bbox_inches="tight")
    return imp


def class_importance(expl, class_index, top=15):
    """Return the largest mean absolute SHAP contributions for one class index."""
    import pandas as pd

    vals = np.abs(expl.values[:, :, class_index]).mean(axis=0)
    display_names = _feature_names(expl, len(vals))
    return (
        pd.DataFrame({"feature": display_names, "importancia": vals})
        .sort_values("importancia", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def class_beeswarm(expl, class_index, max_display=15, plot_title=None, save=None):
    """Plot the SHAP contribution distribution for one pre-calibration class output."""
    import shap
    import matplotlib.pyplot as plt

    e_c = expl[:, :, class_index]
    shap.plots.beeswarm(e_c, max_display=max_display, show=False)
    ax = plt.gca()
    if plot_title:
        ax.set_title(plot_title)
    plt.tight_layout()
    if save is not None:
        plt.gcf().savefig(save, bbox_inches="tight")
    return ax


def class_dependence(
    expl, estimator_model, feature, class_codes, class_names_list=None, logx=True, save=None
):
    """Plot feature values against their pre-calibration SHAP contribution by class."""
    import matplotlib.pyplot as plt

    paleta = [
        OKABE_ITO["naranja"],
        OKABE_ITO["azul"],
        OKABE_ITO["verde"],
        OKABE_ITO["rojo"],
        OKABE_ITO["morado"],
    ]
    j = list(expl.feature_names).index(feature)
    x = np.asarray(expl.data, dtype=float)[:, j]
    clist = list(estimator_model.classes_)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0, color="#666", lw=0.9, zorder=1)
    for n, c in enumerate(class_codes):
        k = clist.index(c)
        lab = class_names_list[k] if class_names_list is not None else str(c)
        ax.scatter(
            x,
            expl.values[:, j, k],
            s=26,
            color=paleta[n % len(paleta)],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
            label=lab,
            zorder=3,
        )
    if logx:
        ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value (positive supports the class)")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return ax


def feature_directions(expl, estimator_model, class_code, top=10, min_support=20):
    """Estimate feature/contribution correlations within supported members of one class."""
    import pandas as pd

    kc = list(estimator_model.classes_).index(class_code)
    V = expl.values[:, :, kc]
    D = np.asarray(expl.data, float)
    display_names = _feature_names(expl, V.shape[1])
    records_list = []
    for j in range(V.shape[1]):
        val, sh = D[:, j], V[:, j]
        m = np.isfinite(val) & np.isfinite(sh)
        imp = np.abs(sh[m]).mean() if m.any() else 0.0
        if m.sum() >= min_support and np.std(val[m]) > 0:
            corr = float(np.corrcoef(val[m], sh[m])[0, 1])
            q = np.quantile(val[m], [0.25, 0.75])
            lo = sh[m & (val <= q[0])].mean()
            hi = sh[m & (val >= q[1])].mean()
        else:
            corr = lo = hi = np.nan
        records_list.append((display_names[j], imp, corr, lo, hi))
    t = (
        pd.DataFrame(
            records_list,
            columns=["feature", "importancia", "corr", "shap_val_bajo", "shap_val_alto"],
        )
        .sort_values("importancia", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )

    def _direction(r):
        if not np.isfinite(r["corr"]):
            return "insufficient support"
        flecha = (
            "high values -> SUPPORT the class"
            if r["shap_val_alto"] > r["shap_val_bajo"]
            else "high values -> OPPOSE the class"
        )
        return ("non-monotonic; " + flecha) if abs(r["corr"]) < 0.15 else flecha

    t["sentido"] = t.apply(_direction, axis=1)
    return t


def direction_table(expl, estimator_model, class_names_list=None, top=8, min_support=20):
    """Combine class-specific feature directions into a table for descriptive interpretation."""
    import pandas as pd

    display_names = (
        class_names_list
        if class_names_list is not None
        else [str(c) for c in estimator_model.classes_]
    )
    chunks_list = []
    for c, nom in zip(estimator_model.classes_, display_names):
        ti = feature_directions(expl, estimator_model, c, top=top, min_support=min_support)
        ti.insert(0, "clase", nom)
        chunks_list.append(ti)
    return pd.concat(chunks_list, ignore_index=True)


def direction_heatmap(
    expl,
    estimator_model,
    class_names_list,
    top=14,
    class_order=None,
    separator_index=None,
    min_support=20,
    save=None,
):
    """Plot supported feature/contribution correlations across the selected class order."""
    import matplotlib.pyplot as plt

    V = expl.values
    D = np.asarray(expl.data, float)
    imp = np.abs(V).mean(axis=(0, 2))
    topf = np.argsort(imp)[::-1][:top]
    nombres_feat = [expl.feature_names[j] for j in topf]
    if class_order is None:
        kidx = list(range(len(class_names_list)))
        class_labels = list(class_names_list)
    else:
        kidx = [list(class_names_list).index(n) for n in class_order]
        class_labels = list(class_order)
    M = np.full((len(topf), len(kidx)), np.nan)
    for a, jf in enumerate(topf):
        val = D[:, jf]
        for b, kc in enumerate(kidx):
            sh = V[:, jf, kc]
            m = np.isfinite(val) & np.isfinite(sh)
            if m.sum() > min_support and np.std(val[m]) > 0:
                M[a, b] = np.corrcoef(val[m], sh[m])[0, 1]
    fig, ax = plt.subplots(figsize=(0.6 * len(kidx) + 2.0, 0.42 * len(topf) + 1.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kidx)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(topf)))
    ax.set_yticklabels(nombres_feat)
    ax.grid(False)
    if separator_index is not None:
        ax.axvline(separator_index - 0.5, color="k", lw=1.2)
    for a in range(len(topf)):
        for b in range(len(kidx)):
            if np.isfinite(M[a, b]):
                ax.text(
                    b,
                    a,
                    f"{M[a, b]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(M[a, b]) > 0.55 else "#222",
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("corr(value, SHAP): positive supports the class")
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return M


def class_support_profile(
    expl, estimator_model, X, es_anomalia, top=15, plot_style="dumbbell", save=None
):
    "Compare negative pre-calibration support for the selected class.\n\nThe selected class comes from the calibrated classifier, but the attributions\nbelong to its underlying forests. This is a descriptive diagnostic, not an\nexact decomposition of the operational novelty function 1 - max(p_calibrated)."
    import pandas as pd
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    es_anomalia = np.asarray(es_anomalia, dtype=bool)
    pred_idx = np.asarray(estimator_model.predict_proba(X)).argmax(axis=1)
    n, f, _ = expl.values.shape
    contrib = -expl.values[np.arange(n), :, pred_idx]
    display_names = _feature_names(expl, f)
    nov_anom = contrib[es_anomalia].mean(axis=0)
    nov_conoc = contrib[~es_anomalia].mean(axis=0)
    result_table = (
        pd.DataFrame(
            {
                "feature": display_names,
                "novedad_anomalias": nov_anom,
                "novedad_conocidos": nov_conoc,
            }
        )
        .assign(diferencia=lambda d: d.novedad_anomalias - d.novedad_conocidos)
        .sort_values("diferencia", ascending=False)
        .reset_index(drop=True)
    )
    d = result_table.head(top)
    y = np.arange(len(d))[::-1]
    if plot_style == "dumbbell":
        fig, ax = plt.subplots(figsize=(7.6, 0.4 * len(d) + 1.2))
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.plot(
                [r.novedad_conocidos, r.novedad_anomalias],
                [yi, yi],
                color="#bbbbbb",
                lw=1.6,
                zorder=1,
            )
        ax.scatter(
            d["novedad_conocidos"],
            y,
            s=48,
            color=OKABE_ITO["azul"],
            zorder=3,
            label="Known classes",
        )
        ax.scatter(
            d["novedad_anomalias"],
            y,
            s=48,
            color=OKABE_ITO["rojo"],
            zorder=3,
            label="Anomalies (class 99)",
        )
        ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper right")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(d) + 1))
        ax.barh(
            y - 0.2,
            d["novedad_anomalias"],
            height=0.4,
            label="anomalies (class 99)",
            color=OKABE_ITO["rojo"],
        )
        ax.barh(
            y + 0.2,
            d["novedad_conocidos"],
            height=0.4,
            label="Known classes",
            color=OKABE_ITO["azul"],
        )
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.axvline(0, color="#666", lw=0.9)
    ax.set_xlabel("Mean negative class support (forests before calibration)")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    if save is not None:
        ax.figure.savefig(save, bbox_inches="tight")
        import hashlib
        import json
        from pathlib import Path

        file_path = Path(save)
        file_path.with_suffix(file_path.suffix + ".meta.json").write_text(
            json.dumps(
                {
                    "semantics": "precalibration_class_support",
                    "figure_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result_table


def explain_object(
    expl,
    i,
    class_index=None,
    estimator_model=None,
    X=None,
    max_display=12,
    plot_title=None,
    save=None,
):
    """Plot a waterfall for one sample and one pre-calibration class output."""
    import shap
    import matplotlib.pyplot as plt

    if class_index is None:
        if estimator_model is None or X is None:
            raise ValueError(
                "Provide class_index, or estimator_model and X to select the predicted class."
            )
        class_index = int(
            np.asarray(estimator_model.predict_proba(np.asarray(X, dtype=float)[[i]])).argmax()
        )
    item_name = None
    if getattr(expl, "class_names", None) is not None:
        item_name = expl.class_names[class_index]
    shap.plots.waterfall(expl[i, :, class_index], max_display=max_display, show=False)
    ax = plt.gca()
    if plot_title:
        ax.set_title(plot_title)
    elif item_name is not None:
        ax.set_title(f"Object #{i} - Contributions to class {item_name}")
    plt.tight_layout()
    if save is not None:
        plt.gcf().savefig(save, bbox_inches="tight")
    return ax


def compare_waterfalls(
    expl, estimator_model, X, sample_positions, plot_titles=None, top=10, save=None
):
    """Compare individual pre-calibration waterfall explanations on a shared feature selection."""
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    proba = np.asarray(estimator_model.predict_proba(X))
    data, allv = [], []
    for pos in sample_positions:
        k = int(proba[pos].argmax())
        sh = expl.values[pos, :, k]
        idx = np.argsort(np.abs(sh))[::-1][:top]
        idx = idx[np.argsort(sh[idx])]
        feats = [expl.feature_names[j] for j in idx]
        data.append((feats, sh[idx]))
        allv.append(np.abs(sh[idx]))
    lim = float(np.max(np.concatenate(allv))) * 1.15
    fig, axes = plt.subplots(1, len(sample_positions), figsize=(5.5 * len(sample_positions), 4.8))
    if len(sample_positions) == 1:
        axes = [axes]
    for a, (ax, (feats, sh)) in enumerate(zip(axes, data)):
        yy = np.arange(len(feats))
        ax.barh(
            yy,
            sh,
            color=[OKABE_ITO["rojo"] if v > 0 else OKABE_ITO["azul"] for v in sh],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_yticks(yy)
        ax.set_yticklabels(feats)
        ax.axvline(0, color="#666", lw=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("SHAP contribution to the selected class")
        if plot_titles is not None:
            ax.set_title(plot_titles[a], fontsize=11)
        ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], marker="s", ls="", ms=9, mfc=OKABE_ITO["rojo"], label="supports the class"
        ),
        Line2D([0], [0], marker="s", ls="", ms=9, mfc=OKABE_ITO["azul"], label="opposes the class"),
    ]
    axes[-1].legend(handles=handles, frameon=False, loc="lower right")
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return axes
