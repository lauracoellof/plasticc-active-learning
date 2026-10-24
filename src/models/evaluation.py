"""Classification, calibration and paired-seed comparison metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, log_loss


def evaluate(item_name, estimator_model, X_tr, y_tr, X_te, y_te, result_records, notas=""):
    """Fit one estimator on a fixed training split and append its held-out classification metrics."""
    estimator_model.fit(X_tr, y_tr)
    y_pred = estimator_model.predict(X_te)
    fila = {
        "variante": item_name,
        "WF1": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "MF1": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "notas": notas,
    }
    result_records.append(fila)
    print(
        f"{item_name:32s}  WF1={fila['WF1']:.4f}  MF1={fila['MF1']:.4f}  "
        f"acc={fila['accuracy']:.4f}"
    )
    return estimator_model, y_pred


def cross_validated_predictions(estimator_model, X, y, n_folds=5, seed_value=42):
    """Return out-of-fold predicted labels from shuffled stratified cross-validation."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed_value)
    return cross_val_predict(estimator_model, X, y, cv=skf, n_jobs=-1)


def _class_indices(class_codes):
    """Map the ordered class codes to their probability-array column indices."""
    return {c: i for i, c in enumerate(class_codes)}


def balanced_log_loss(y_true, proba, class_codes):
    """Average per-class mean negative log likelihood, giving equal weight to represented classes."""
    y_true = np.asarray(y_true)
    P = np.clip(proba, 1e-15, 1 - 1e-15)
    col = _class_indices(class_codes)
    perdidas = [-np.log(P[y_true == c, col[c]]).mean() for c in class_codes if (y_true == c).any()]
    return float(np.mean(perdidas))


def multiclass_brier_score(y_true, proba, class_codes):
    "Mean squared distance from the true one-hot vector; sum over classes without dividing by the number of classes."
    y_true = np.asarray(y_true)
    col = _class_indices(class_codes)
    Y = np.zeros_like(proba)
    for i, yt in enumerate(y_true):
        Y[i, col[yt]] = 1.0
    return float(((proba - Y) ** 2).sum(axis=1).mean())


def confidence_ece(y_true, proba, class_codes, n_bins=10):
    "Return top-label ECE and its reliability curve.\n\nUse equally spaced confidence bins and weight each absolute difference between\nmean confidence and empirical accuracy by its fraction of the population."
    y_true = np.asarray(y_true)
    class_codes = np.asarray(class_codes)
    conf = proba.max(axis=1)
    pred = class_codes[proba.argmax(axis=1)]
    acierto = (pred == y_true).astype(float)
    bordes = np.linspace(0, 1, n_bins + 1)
    ece, N, light_curve = 0.0, len(y_true), []
    for b in range(n_bins):
        m = (conf > bordes[b]) & (conf <= bordes[b + 1])
        if m.sum() == 0:
            continue
        acc, cf = acierto[m].mean(), conf[m].mean()
        ece += (m.sum() / N) * abs(acc - cf)
        light_curve.append((cf, acc, int(m.sum())))
    return float(ece), light_curve


def probability_summary(item_name, y_true, proba, class_codes):
    """Print probability metrics and return the top-label reliability-curve bins."""
    ll = log_loss(y_true, proba, labels=list(class_codes))
    llb = balanced_log_loss(y_true, proba, class_codes)
    br = multiclass_brier_score(y_true, proba, class_codes)
    ece, light_curve = confidence_ece(y_true, proba, class_codes)
    print(
        f"{item_name:22s}  logloss={ll:.4f}  logloss_bal={llb:.4f}  brier={br:.4f}  ECE={ece:.4f}"
    )
    return light_curve


def paired_comparison_summary(
    proposed_results, reference_results, metricas, pair_key="semilla", n_boot=20_000, seed_value=42
):
    "Compare matched runs using paired differences.\n\nReturn mean differences, bootstrap intervals, win counts, rank-biserial effect\nsizes and two-sided Wilcoxon tests with Holm adjustment across the requested\nmetrics. Intervals quantify Monte Carlo variability across seeds on this pool,\nnot uncertainty about the future Rubin population."
    from scipy.stats import rankdata, wilcoxon

    def _by_key(df, item_name):
        if pair_key in df.columns:
            if df[pair_key].duplicated().any():
                raise ValueError(f"{item_name} contains duplicate values of {pair_key!r}")
            return df.set_index(pair_key)
        if df.index.name == pair_key:
            if df.index.duplicated().any():
                raise ValueError(f"{item_name} contains duplicate values of {pair_key!r}")
            return df
        raise ValueError(f"{item_name} does not contain the pairing key {pair_key!r}")

    proposed_results = _by_key(proposed_results, "propuesto")
    reference_results = _by_key(reference_results, "referencia")
    if set(proposed_results.index) != set(reference_results.index):
        raise ValueError("The systems do not contain exactly the same repetitions")

    metricas = list(metricas)
    rng = np.random.default_rng(seed_value)
    records_list = []
    for metric_name in metricas:
        if metric_name not in proposed_results or metric_name not in reference_results:
            raise ValueError(f"Metric {metric_name!r} is missing from one of the systems")
        a = proposed_results.loc[sorted(proposed_results.index), metric_name].to_numpy(dtype=float)
        b = reference_results.loc[sorted(reference_results.index), metric_name].to_numpy(
            dtype=float
        )
        paired_difference = a - b
        if not np.isfinite(paired_difference).all():
            raise ValueError(f"Metric {metric_name!r} contains non-finite paired differences")

        sample_records = paired_difference[
            rng.integers(0, len(paired_difference), size=(n_boot, len(paired_difference)))
        ].mean(axis=1)
        no_cero = paired_difference[paired_difference != 0]
        if len(no_cero):
            signed_ranks = rankdata(np.abs(no_cero))
            rango_biserial = (
                signed_ranks[no_cero > 0].sum() - signed_ranks[no_cero < 0].sum()
            ) / signed_ranks.sum()
            method_name = "exact" if len(no_cero) == len(paired_difference) else "approx"
            p_valor = wilcoxon(
                paired_difference, alternative="two-sided", method=method_name
            ).pvalue
        else:
            rango_biserial = 0.0
            p_valor = 1.0

        records_list.append(
            {
                "metrica": metric_name,
                "n": len(paired_difference),
                "media_propuesto": a.mean(),
                "media_referencia": b.mean(),
                "diferencia_media": paired_difference.mean(),
                "ic95_inf": np.quantile(sample_records, 0.025),
                "ic95_sup": np.quantile(sample_records, 0.975),
                "diferencia_mediana": np.median(paired_difference),
                "victorias": int((paired_difference > 0).sum()),
                "rango_biserial": float(rango_biserial),
                "p_wilcoxon": float(p_valor),
            }
        )

    summary_table = pd.DataFrame(records_list)
    sort_order = np.argsort(summary_table["p_wilcoxon"].to_numpy())
    p_ordenados = summary_table.loc[sort_order, "p_wilcoxon"].to_numpy()
    m = len(p_ordenados)
    p_ajustados = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p_ordenados))
    summary_table["p_holm"] = np.nan
    summary_table.loc[sort_order, "p_holm"] = p_ajustados
    return summary_table
