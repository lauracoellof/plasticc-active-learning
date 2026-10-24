"""Finite-population diagnostics, Monte Carlo repeatability and Holm adjustment."""

import numpy as np
import pytest

from src.models.subsampling import holm_adjustment, evaluate_subsampling, load_validation


def test_holm_adjustment_preserves_input_order():
    assert np.allclose(holm_adjustment([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])


def test_subsampling_is_reproducible_and_identifies_medians():
    conf = np.array([0.1, 0.2, 0.3, 0.9, 0.15, 0.25, 0.35, 0.95])
    labels = np.array([6, 991, 6, 992, 6, 991, 6, 992])
    sub = np.array([True] * 4 + [False] * 4)
    kwargs = dict(isolation=np.arange(8), residuos=np.arange(8) ** 2, n_remuestreos=20)
    tests, summary, samples = evaluate_subsampling(conf, labels, sub, **kwargs)
    t2, s2, m2 = evaluate_subsampling(conf, labels, sub, **kwargs)
    assert tests.equals(t2) and summary.equals(s2) and samples.equals(m2)
    row = tests.loc[tests.magnitud.eq("confidence (gate)")].iloc[0]
    assert row.resumen == "mediana"
    assert row.submuestreado == pytest.approx(0.25)
    assert np.isnan(tests.loc[tests.magnitud.eq("novelty 1-conf"), "p_ajustado_Holm"].iloc[0])


def test_empty_groups_or_misaligned_vectors_are_rejected():
    with pytest.raises(ValueError, match="aligned"):
        evaluate_subsampling([0.2, 0.3], [6], [True, False], isolation=[0, 1], residuos=[0, 1])
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_subsampling([0.2, 0.3], [6, 991], [True, True], isolation=[0, 1], residuos=[0, 1])


def test_historical_tables_require_provenance(tmp_path):
    (tmp_path / "validacion_submuestreo_tests.csv").write_text("magnitud,p_valor\nx,0.1\n")
    with pytest.raises(RuntimeError, match="without provenance"):
        load_validation(tmp_path)
