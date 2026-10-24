import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import paired_comparison_summary


def test_paired_summary_reports_effect_sizes_and_holm_adjustment():
    reference_results = pd.DataFrame(
        {
            "semilla": range(10),
            "MF1": np.linspace(0.60, 0.69, 10),
            "recall_anom": np.linspace(0.010, 0.019, 10),
        }
    )
    proposed_results = reference_results.copy()
    proposed_results["MF1"] += 0.03
    proposed_results["recall_anom"] += 0.01

    summary_table = paired_comparison_summary(
        proposed_results, reference_results, ["MF1", "recall_anom"], n_boot=2_000
    ).set_index("metrica")

    assert (summary_table["victorias"] == 10).all()
    assert np.allclose(summary_table["rango_biserial"], 1.0)
    assert summary_table.loc["MF1", "diferencia_media"] == pytest.approx(0.03)
    assert summary_table.loc["recall_anom", "diferencia_media"] == pytest.approx(0.01)
    assert (summary_table["ic95_inf"] > 0).all()
    assert (summary_table["p_holm"] >= summary_table["p_wilcoxon"]).all()
    assert (summary_table["p_holm"] < 0.05).all()


def test_paired_comparison_requires_matching_seeds():
    proposed_results = pd.DataFrame({"semilla": [1, 2], "MF1": [0.7, 0.7]})
    reference_results = pd.DataFrame({"semilla": [1, 3], "MF1": [0.6, 0.6]})

    with pytest.raises(ValueError, match="same repetitions"):
        paired_comparison_summary(proposed_results, reference_results, ["MF1"], n_boot=100)
