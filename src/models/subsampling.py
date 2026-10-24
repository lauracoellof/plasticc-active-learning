"""Finite-population subsample diagnostics and verified result loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, norm
from sklearn.metrics import roc_auc_score


def load_validation(directory_path):
    "Load validation tables only when their bytes match the execution manifest."
    import hashlib
    import json
    from pathlib import Path

    root = Path(directory_path)
    manifest_path = root / "validacion_submuestreo.manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "Validation without provenance: run scripts/regenerate_subsampling_validation.py in the reference environment"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = []
    for suffix in ["tests", "remuestreo", "realizaciones"]:
        name = f"validacion_submuestreo_{suffix}.csv"
        path = root / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[
            "outputs_sha256"
        ].get(name):
            raise RuntimeError(f"Missing or modified validation table: {name}")
        tables.append(pd.read_csv(path))
    return (*tables, manifest)


def holm_adjustment(p_values):
    """Return Holm-adjusted p-values in the original hypothesis order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("P-values must be finite and lie between zero and one")
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return result


def _proportions(x1, n1, x2, n2):
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (x1 / n1 - x2 / n2) / se
    return float(z), float(2 * norm.sf(abs(z)))


def evaluate_subsampling(
    confianza,
    class_labels,
    sub,
    *,
    isolation,
    residuos,
    n_remuestreos=200,
    seed_value=42,
    threshold_value=0.30,
):
    "Compare a subsample with its disjoint finite-population complement.\n\nReport medians, exploratory KS/proportion/composition tests and Holm-adjusted\np-values. Exclude novelty from Holm because it equals one minus confidence.\nUse Monte Carlo samples without replacement, not bootstrap samples. Preserve\nfull precision until presentation. Undefined one-class ROC-AUC draws are\nrecorded as missing and the number of valid draws is reported."
    conf = np.asarray(confianza, dtype=float)
    tt = np.asarray(class_labels)
    sub = np.asarray(sub, dtype=bool)
    iso, resid = np.asarray(isolation, dtype=float), np.asarray(residuos, dtype=float)
    if any(v.ndim != 1 or len(v) != len(conf) for v in [conf, tt, sub, iso, resid]):
        raise ValueError("All vectors must be one-dimensional and aligned")
    if not all(np.isfinite(v).all() for v in [conf, tt, iso, resid]):
        raise ValueError("Input vectors contain non-finite values")
    if ((conf < 0) | (conf > 1)).any() or not 0 < threshold_value < 1:
        raise ValueError("Confidence values and threshold must lie in [0, 1]")
    n1, n2 = int(sub.sum()), int((~sub).sum())
    if not n1 or not n2 or n_remuestreos < 2:
        raise ValueError("Two non-empty groups and at least two resamples are required")
    y = (tt >= 990).astype(int)
    if len(np.unique(y[sub])) != 2 or len(np.unique(y[~sub])) != 2:
        raise ValueError("Both groups must contain known objects and anomalies")
    records = []
    for name, values in [
        ("confidence (gate)", conf),
        ("novelty 1-conf", 1 - conf),
        ("Isolation Forest", iso),
        ("residuals GP", resid),
    ]:
        stat, p = ks_2samp(values[sub], values[~sub])
        records.append(
            dict(
                magnitud=name,
                test="KS",
                estadístico=stat,
                p_valor=p,
                submuestreado=float(np.median(values[sub])),
                complemento=float(np.median(values[~sub])),
                resumen="mediana",
            )
        )
    for name, event in [
        ("tasa cand. conf<0.30", conf < threshold_value),
        ("prevalencia anomalies 99x", y.astype(bool)),
    ]:
        a, b = int(event[sub].sum()), int(event[~sub].sum())
        stat, p = _proportions(a, n1, b, n2)
        records.append(
            dict(
                magnitud=name,
                test="z prop.",
                estadístico=stat,
                p_valor=p,
                submuestreado=100 * a / n1,
                complemento=100 * b / n2,
                resumen="porcentaje",
            )
        )
    for name, mask in [
        ("known-class composition", tt < 990),
        ("99x subtype composition", tt >= 990),
    ]:
        table = pd.crosstab(tt[mask], sub[mask])
        stat, p, _, _ = chi2_contingency(table)
        records.append(
            dict(
                magnitud=name,
                test="chi2",
                estadístico=stat,
                p_valor=p,
                submuestreado=int(mask[sub].sum()),
                complemento=int(mask[~sub].sum()),
                resumen="recuento",
            )
        )
    tests = pd.DataFrame(records)
    independent = tests.magnitud.ne("novelty 1-conf")
    tests["p_ajustado_Holm"] = np.nan
    tests.loc[independent, "p_ajustado_Holm"] = holm_adjustment(tests.loc[independent, "p_valor"])
    rng = np.random.default_rng(seed_value)
    rates, aucs = [], []
    for _ in range(n_remuestreos):
        idx = rng.choice(len(conf), n1, replace=False)
        rates.append(100 * np.mean(conf[idx] < threshold_value))

        aucs.append(roc_auc_score(y[idx], 1 - conf[idx]) if len(np.unique(y[idx])) == 2 else np.nan)
    samples = pd.DataFrame({"tasa_candidatos_pct": rates, "roc_auc": aucs})
    summary = []
    for name, values, full, used in [
        (
            "tasa candidatos conf<0.30 (%)",
            np.asarray(rates),
            100 * np.mean(conf < threshold_value),
            100 * np.mean(conf[sub] < threshold_value),
        ),
        (
            "ROC-AUC novelty 99x",
            np.asarray(aucs),
            roc_auc_score(y, 1 - conf),
            roc_auc_score(y[sub], 1 - conf[sub]),
        ),
    ]:
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError("Too few resamples have a defined ROC-AUC")
        low, high = np.quantile(values, [0.025, 0.975])
        summary.append(
            dict(
                magnitud=name,
                batch_completo=full,
                submuestra_utilizada=used,
                media_remuestreo=values.mean(),
                desv_remuestreo=values.std(ddof=0),
                ic95_inferior=low,
                ic95_superior=high,
                percentil_submuestra=100 * np.mean(values <= used),
                n_remuestreos_validos=len(values),
            )
        )
    return tests, pd.DataFrame(summary), samples
