"""Classifier representation contracts and compatible saved-model loading."""

import numpy as np
import pandas as pd

from src.models import classification


def test_e2gp_preserves_the_canonical_specification():
    y = np.repeat([10, 20, 30], 3)
    estimator_model = classification.e2gp_al(y=y, random_state=17)
    model_parameters = estimator_model.get_params(deep=True)

    assert estimator_model.method == "sigmoid"
    assert estimator_model.cv == 3
    assert model_parameters["estimator__n_estimators"] == 150
    assert model_parameters["estimator__min_samples_leaf"] == 3
    assert model_parameters["estimator__class_weight"] == "balanced"
    assert model_parameters["estimator__random_state"] == 17


def test_calibration_folds_adapt_to_small_classes():
    estimator_model = classification.e2gp_al(y=[10, 10, 20, 20], random_state=3)
    assert estimator_model.cv == 2


def test_test_matrix_uses_training_column_order():
    train = pd.DataFrame(
        {
            "object_id": [1, 2],
            "target": [10, 20],
            "feature_b": [3.0, 4.0],
            "feature_a": [1.0, np.nan],
            "comentario": ["uno", "dos"],
        }
    )
    X_train, feature_columns = classification.e2gp_matrix(train)

    test = pd.DataFrame(
        {
            "feature_a": [8.0, np.nan],
            "object_id": [3, 4],
            "feature_b": [6.0, 7.0],
            "columna_ajena": [100.0, 200.0],
        }
    )
    X_test, columnas_test = classification.e2gp_matrix(test, feature_columns=feature_columns)

    assert feature_columns == ["feature_b", "feature_a"]
    assert columnas_test == feature_columns
    np.testing.assert_allclose(X_train, [[3.0, 1.0], [4.0, 0.0]])
    np.testing.assert_allclose(X_test, [[6.0, 8.0], [7.0, 0.0]])


def test_model_artifact_and_contract_roundtrip(tmp_path):
    y = np.repeat([10, 20, 30], 6)
    X = np.arange(54, dtype=float).reshape(18, 3)
    estimator_model = classification.calibrated_al_classifier(
        y=y, n_estimators=3, cv=3, random_state=5, n_jobs=1
    ).fit(X, y)
    ruta_modelo = tmp_path / "modelo.joblib"
    ruta_meta = tmp_path / "modelo.meta.json"
    ruta_hash = tmp_path / "checksums.sha256"

    metadata_table = classification.save_e2gp_artifact(
        estimator_model,
        ["a", "b", "c"],
        n_train=len(y),
        ruta_modelo=ruta_modelo,
        ruta_metadatos=ruta_meta,
        ruta_checksums=ruta_hash,
    )
    cargado, feature_columns, meta_cargada = classification.load_e2gp_artifact(
        ruta_modelo=ruta_modelo, ruta_metadatos=ruta_meta
    )

    assert feature_columns == ["a", "b", "c"]
    assert meta_cargada == metadata_table
    assert metadata_table["n_train"] == 18
    assert metadata_table["n_features"] == 3
    np.testing.assert_allclose(cargado.predict_proba(X), estimator_model.predict_proba(X))
