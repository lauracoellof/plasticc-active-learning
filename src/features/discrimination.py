"""Exploratory standardized effect sizes for individual photometric features."""

import numpy as np
import pandas as pd

from src.features.statistics import NEW_FEATURES, COLS_META


def cohens_d(x1, x2):
    """Return the signed standardized mean difference using the pooled sample variance.

    Return NaN for groups with fewer than two rows and zero for zero pooled variance."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    s_pool = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pool == 0:
        return 0.0
    return (x1.mean() - x2.mean()) / s_pool


def feature_block(feature):
    """Map a feature name to the historical statistical, metadata or cross-band block."""
    if feature in NEW_FEATURES:
        return "espectro-temporal"
    if feature in COLS_META:
        return "metadato"
    return "estadistica"


def _feature_columns(result_table, col_target, exclude_values):
    return [c for c in result_table.columns if c not in exclude_values and c != col_target]


def between_class_effect_sizes(
    result_table, class_a, class_b, col_target="target", exclude_values=("object_id",)
):
    """Rank numeric features by absolute Cohen effect size between two labelled classes."""
    cols = _feature_columns(result_table, col_target, exclude_values)
    a = result_table[result_table[col_target] == class_a]
    b = result_table[result_table[col_target] == class_b]
    records_list = [{"feature": c, "d": cohens_d(a[c], b[c])} for c in cols]
    out = pd.DataFrame(records_list)
    out["abs_d"] = out["d"].abs()
    out["bloque"] = out["feature"].map(feature_block)
    return out.sort_values("abs_d", ascending=False).reset_index(drop=True)


def one_vs_rest_effect_sizes(
    result_table, class_code, col_target="target", exclude_values=("object_id",)
):
    """Rank numeric features by absolute Cohen effect size for one class versus the rest."""
    cols = _feature_columns(result_table, col_target, exclude_values)
    a = result_table[result_table[col_target] == class_code]
    remaining_values = result_table[result_table[col_target] != class_code]
    records_list = [{"feature": c, "d": cohens_d(a[c], remaining_values[c])} for c in cols]
    out = pd.DataFrame(records_list)
    out["abs_d"] = out["d"].abs()
    out["bloque"] = out["feature"].map(feature_block)
    return out.sort_values("abs_d", ascending=False).reset_index(drop=True)


def block_summary(result_table, pares, col_target="target", exclude_values=("object_id",)):
    """Summarize mean absolute feature effect sizes for each requested class pair."""
    series = []
    for a, b in pares:
        tab = between_class_effect_sizes(
            result_table, a, b, col_target=col_target, exclude_values=exclude_values
        )
        m = tab.groupby("bloque")["abs_d"].mean()
        m.name = f"{a} vs {b}"
        series.append(m)
    return pd.concat(series, axis=1).round(3)
