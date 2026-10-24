"""Unsupervised anomaly descriptors, rank fusion and retrieval diagnostics."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def rank01(x):
    """Return ordinal ranks scaled to [0, 1]; tied values retain the historical argsort ordering.

    This is not an average-rank transform. Tie ordering may vary across NumPy
    builds, so exact historical fusion results require the declared environment."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return np.argsort(np.argsort(x)) / (n - 1) if n > 1 else np.zeros(n)


def robust_zscore(x):
    """Center by the median and scale by IQR/1.349, with a standard-deviation fallback."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    esc = iqr / 1.349 if iqr > 0 else (np.nanstd(x) or 1.0)
    return np.nan_to_num((x - med) / esc, nan=0.0)


def score_isolation_forest(X, contamination_value=0.02, seed_value=42):
    """Return negative IsolationForest sample scores so larger values indicate greater novelty."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    det = IsolationForest(contamination=contamination_value, random_state=seed_value).fit(X)
    return -det.score_samples(X)


_RESIDUAL_PREFIXES = ("gp_chi2_", "gp_resid_std_", "gp_resid_max_", "gp_outfrac_")


def gp_residual_score(df_gp, prefixes_list=_RESIDUAL_PREFIXES):
    """Average robustly scaled GP residual descriptors; reject missing residual columns."""
    cols = [c for c in df_gp.columns if any(c.startswith(p) for p in prefixes_list)]
    if not cols:
        raise ValueError(f"No GP residual columns match the prefixes {prefixes_list}.")
    Z = np.column_stack([robust_zscore(df_gp[c].to_numpy()) for c in cols])
    return Z.mean(axis=1)


def confidence_novelty_score(proba):
    """Return one minus the calibrated maximum class probability for each object."""
    return 1.0 - np.max(np.asarray(proba, dtype=float), axis=1)


def amplitude_score(df_stats, band_names=("u", "g", "r", "i", "z", "y")):
    """Return a robustly scaled maximum observed flux range across available bands."""
    ampl = []
    for b in band_names:
        cmax, cmin = f"max_{b}", f"min_{b}"
        if cmax in df_stats.columns and cmin in df_stats.columns:
            ampl.append((df_stats[cmax] - df_stats[cmin]).to_numpy(dtype=float))
    if not ampl:
        raise ValueError("No max_b/min_b columns were found in df_stats.")
    return robust_zscore(np.nanmax(np.column_stack(ampl), axis=1))


def fuse_scores(scores, score_weights=None, method_name="rank"):
    """Normalize aligned descriptor arrays and compute their weighted mean."""
    norm = rank01 if method_name == "rank" else robust_zscore
    display_names = list(scores)
    M = np.column_stack([norm(scores[k]) for k in display_names])
    w = (
        np.ones(len(display_names))
        if score_weights is None
        else np.array([score_weights.get(k, 0.0) for k in display_names], dtype=float)
    )
    if w.sum() == 0:
        raise ValueError("The sum of weights is zero.")
    return (M * w).sum(axis=1) / w.sum()


def precision_at_k(y_bin, score, k):
    """Return the anomaly fraction among the k highest-scoring objects."""
    top = np.argsort(np.asarray(score))[::-1][:k]
    return float(np.asarray(y_bin)[top].mean())


def evaluate_scores(scores, y_bin, ks=(50, 100, 200), incluir_fusion=True, score_weights=None):
    """Summarize anomaly ROC-AUC and precision-at-k, optionally including rank fusion."""
    from sklearn.metrics import roc_auc_score

    y_bin = np.asarray(y_bin)
    todos = dict(scores)
    if incluir_fusion:
        todos["FUSION"] = fuse_scores(scores, score_weights=score_weights)
    records_list = []
    for item_name, s in todos.items():
        fila = {"descriptor": item_name, "ROC_AUC": round(roc_auc_score(y_bin, s), 4)}
        for k in ks:
            fila[f"P@{k}"] = round(precision_at_k(y_bin, s, k), 4)
        records_list.append(fila)
    return pd.DataFrame(records_list).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)


def batch_descriptors(batch, clf, feature_columns, meta_glob, dir_procesado=None):
    """Load aligned batch feature caches and compute descriptors against a fixed classifier."""
    from pathlib import Path
    import config
    from src.models import classification
    from src.features import statistics as est
    from src.features.cache import load_cache

    dirp = Path(dir_procesado) if dir_procesado is not None else config.DIR_PROCESSED
    suf = "" if int(batch) == 1 else f"_b{int(batch)}"
    fs = dirp / f"features_estadisticas_test{suf}.csv"
    fg = dirp / f"features_gp_test{suf}.csv"
    fg2 = dirp / f"features_gp2d_test{suf}.csv"

    df = est.add_derived_features(
        load_cache(fs, "estadisticas")
        .merge(load_cache(fg, "gp1d"), on="object_id")
        .merge(load_cache(fg2, "gp2d"), on="object_id")
    )
    X, _ = classification.e2gp_matrix(df, feature_columns)
    gp = df[["object_id"]].merge(load_cache(fg, "gp1d"), on="object_id", how="left")
    st = df[["object_id"]].merge(load_cache(fs, "estadisticas"), on="object_id", how="left")
    tt = df[["object_id"]].merge(meta_glob, on="object_id", how="left")["true_target"].to_numpy()

    scores = {
        "IsolationForest": score_isolation_forest(X),
        "residuos_GP": gp_residual_score(gp),
        "novedad_1-conf": confidence_novelty_score(clf.predict_proba(X)),
        "cambios_mag": amplitude_score(st),
    }
    return scores, (tt >= 990).astype(int), tt
