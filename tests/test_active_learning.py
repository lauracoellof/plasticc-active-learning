"""Oracle-label routing, query budgets, acquisition scores and matched experiment splits."""

import numpy as np
import pandas as pd
import pytest

from src.models import active_learning as al
from src.models import anomalies as an


class _DeterministicClassifier:
    """Oracle-label routing, query budgets, acquisition scores and matched experiment splits."""

    def __init__(self, recorded_rows):
        self.recorded_rows = recorded_rows

    def fit(self, X, y):
        y = np.asarray(y).copy()
        self.recorded_rows.append(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise AssertionError("The synthetic scenario must retain two known classes")
        return self

    def predict_proba(self, X):

        return np.full((len(X), 2), 0.5, dtype=float)

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def _synthetic_scenario():
    X_tr = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]])
    y_tr = np.array([10, 10, 20, 20])
    X_pool = np.arange(16, dtype=float).reshape(8, 2)
    y_pool = np.array([991, 10, 20, 992, 10, 20, 10, 20])
    X_eval = np.array([[0.2, 0.1], [0.3, 0.2], [1.2, 1.1], [1.3, 1.2]])
    y_eval = np.array([10, 10, 20, 20])
    desc_pool = np.column_stack(
        [
            np.linspace(0.0, 1.0, len(X_pool)),
            np.linspace(1.0, 0.0, len(X_pool)),
        ]
    )
    return X_tr, y_tr, X_pool, y_pool, desc_pool, X_eval, y_eval


def _model_factory(recorded_rows):
    def make_model(y_actual):

        assert np.all(np.asarray(y_actual) < 990)
        return _DeterministicClassifier(recorded_rows)

    return make_model


@pytest.mark.parametrize("method_name", ["entropy", "margin", "least_confidence"])
def test_uniform_probabilities_have_higher_uncertainty(method_name):
    proba = np.array([[0.5, 0.5], [0.99, 0.01]])
    score = al.uncertainty_score(proba, method_name=method_name)
    assert score[0] > score[1]


def test_uncertainty_rejects_unknown_methods():
    with pytest.raises(ValueError, match="unknown method"):
        al.uncertainty_score([[0.5, 0.5]], method_name="inventado")


def test_al_reuses_canonical_anomaly_descriptors():
    assert al._rank01 is an.rank01
    assert al._zscore_robusto is an.robust_zscore
    assert al.gp_residual_score is an.gp_residual_score


def test_multiseed_split_holds_out_known_objects_and_keeps_anomalies_in_pool():
    y = np.array([10] * 8 + [20] * 8 + [991, 992])
    split_records = list(
        al._iter_test_splits(y, random_seeds=(3, 7), test_size_eval=0.25, etq_anom_min=990)
    )

    assert [seed_value for seed_value, _, _ in split_records] == [3, 7]
    anomalies = set(np.where(y >= 990)[0])
    known_objects = set(np.where(y < 990)[0])
    for _, pool, evaluation in split_records:
        assert set(evaluation).issubset(known_objects)
        assert anomalies.issubset(set(pool))
        assert set(pool).isdisjoint(set(evaluation))
        assert (set(pool) & known_objects) | set(evaluation) == known_objects


def test_multiseed_output_is_incremental_without_duplicate_headers(tmp_path):
    file_path = tmp_path / "resultados.csv"
    file_path.write_text("salida previous", encoding="utf-8")
    al._reset_multiseed_output(file_path)
    assert not file_path.exists()

    acumulados = []
    al._record_multiseed_result(
        acumulados,
        pd.DataFrame({"iter": [0], "MF1": [0.5]}),
        file_path,
        seed_value=1,
        run_mode="triage",
    )
    al._record_multiseed_result(
        acumulados,
        pd.DataFrame({"iter": [0], "MF1": [0.6]}),
        file_path,
        seed_value=2,
        run_mode="triage",
    )

    expected_value_value = pd.concat(acumulados, ignore_index=True)
    observado = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(observado, expected_value_value)
    assert file_path.read_text(encoding="utf-8").count("iter,MF1,semilla,modo") == 1


def test_quota_selection_respects_gate_without_duplicates():
    pool = np.arange(10, 16)
    candidatas = np.array([True, True, False, False, False, True])
    seleccion = al.select_by_quota(
        pool_idx=pool,
        s_unc=np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3]),
        mask_candidata=candidatas,
        s_nov=np.array([0.8, 0.9, 0.2, 0.1, 0.3, 0.7]),
        b_cls=2,
        b_nov=2,
    )

    assert set(seleccion["anomalia"]) == {10, 11}
    assert set(seleccion["clasificacion"]) == {12, 13}
    assert set(seleccion["anomalia"]).isdisjoint(seleccion["clasificacion"])


def test_triage_respects_budget_and_excludes_anomalies_from_training():
    X_tr, y_tr, X_pool, y_pool, _, X_eval, y_eval = _synthetic_scenario()
    recorded_rows = []
    hist = al.confidence_triage(
        X_tr,
        y_tr,
        X_pool,
        y_pool,
        X_eval,
        y_eval,
        _model_factory(recorded_rows),
        conf_anom=0.55,
        conf_dudoso=0.90,
        budget=5,
        b_iter=2,
        n_iter=5,
        random_state=7,
    )

    assert hist["n_consultas"].max() <= 5
    assert hist.iloc[-1]["n_consultas"] == 5
    assert all(np.all(y_fit < 990) for y_fit in recorded_rows)


def test_random_triage_is_reproducible_with_the_same_seed():
    X_tr, y_tr, X_pool, y_pool, _, X_eval, y_eval = _synthetic_scenario()

    def run():
        recorded_rows = []
        hist = al.confidence_triage(
            X_tr,
            y_tr,
            X_pool,
            y_pool,
            X_eval,
            y_eval,
            _model_factory(recorded_rows),
            run_mode="random",
            budget=6,
            b_iter=2,
            n_iter=5,
            random_state=19,
        )
        return hist, recorded_rows

    hist_1, registros_1 = run()
    hist_2, registros_2 = run()

    pd.testing.assert_frame_equal(hist_1, hist_2)
    assert len(registros_1) == len(registros_2)
    assert all(np.array_equal(a, b) for a, b in zip(registros_1, registros_2))


def test_two_channels_respect_total_and_channel_budgets():
    X_tr, y_tr, X_pool, y_pool, desc_pool, X_eval, y_eval = _synthetic_scenario()
    recorded_rows = []
    hist = al.two_channel_pipeline(
        X_tr,
        y_tr,
        X_pool,
        y_pool,
        desc_pool,
        X_eval,
        y_eval,
        _model_factory(recorded_rows),
        frac_desc=0.4,
        conf_dudoso=0.90,
        budget=5,
        b_iter=2,
        n_iter=5,
        random_state=23,
    )

    final = hist.iloc[-1]
    assert hist["n_consultas"].max() <= 5
    assert final["n_consultas"] == 5
    assert final["cons_desc"] <= 2
    assert final["cons_clas"] <= 3
    assert final["n_consultas"] == final["cons_desc"] + final["cons_clas"]
    assert all(np.all(y_fit < 990) for y_fit in recorded_rows)
