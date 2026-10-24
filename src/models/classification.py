"""Random-Forest variants and the canonical calibrated E2-GP model contract."""

import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier


CLASS_NAMES = {
    6: "uLens",
    15: "TDE",
    16: "EB",
    42: "SNII",
    52: "SNIax",
    53: "Mira",
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    92: "RRL",
    95: "SLSN-I",
}


SUPERNOVA_CLASSES = (42, 52, 62, 67, 90)


SUPERNOVA_LABEL = -1


def _default_forest(random_state=42):
    """Construct the uncalibrated 150-tree class-balanced comparison forest."""
    return RandomForestClassifier(
        n_estimators=150, class_weight="balanced", random_state=random_state, n_jobs=-1
    )


def group_supernovae(y, supernova_classes=SUPERNOVA_CLASSES, supernova_label=SUPERNOVA_LABEL):
    """Replace the selected supernova labels with one shared top-level class code."""
    y = np.asarray(y)
    return np.where(np.isin(y, list(supernova_classes)), supernova_label, y)


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """Classify a supernova aggregate first, then resolve its subtypes with a second forest."""

    def __init__(self, supernova_classes=SUPERNOVA_CLASSES, base_estimator=None, random_state=42):

        self.supernova_classes = supernova_classes
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X, y):
        "Fit."
        X = np.asarray(X)
        y = np.asarray(y)
        base = (
            self.base_estimator
            if self.base_estimator is not None
            else _default_forest(self.random_state)
        )

        y_top = group_supernovae(y, self.supernova_classes, SUPERNOVA_LABEL)
        self.clf_top_ = clone(base).fit(X, y_top)

        mask = np.isin(y, list(self.supernova_classes))
        self.clf_sub_ = clone(base).fit(X[mask], y[mask])
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        "Predict."
        X = np.asarray(X)
        pred = self.clf_top_.predict(X).astype(self.clf_sub_.classes_.dtype)
        idx = np.where(pred == SUPERNOVA_LABEL)[0]
        if idx.size:
            pred[idx] = self.clf_sub_.predict(X[idx])
        return pred


def calibrated_al_classifier(
    y=None,
    n_estimators=150,
    min_samples_leaf=3,
    method_name="sigmoid",
    cv=3,
    random_state=42,
    n_jobs=-1,
):
    "Build a class-balanced Random Forest with sigmoid calibration.\n\nThe default configuration uses 150 trees, a minimum leaf size of three and\nthree internal calibration folds. When labels are supplied, the fold count is\nreduced for small classes. Refit and recalibrate after each AL acquisition."
    from sklearn.calibration import CalibratedClassifierCV

    base = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    cv_ef = cv
    if y is not None:
        _, class_counts = np.unique(np.asarray(y), return_counts=True)
        cv_ef = max(2, min(cv, int(class_counts.min())))
    return CalibratedClassifierCV(base, method=method_name, cv=cv_ef)


def features_e2gp(split="train", *, batch=None):
    """Load and merge the three validated feature families, then append derived features.

    Preserve column order: the saved estimator contract depends on the ordered
    feature schema, not only on the set of feature names."""
    import config
    from src.features import statistics as est
    from src.features.cache import load_cache

    if split == "train":
        fs, fg, fg2 = config.FEATURES_CSV, config.FEATURES_GP_CSV, config.FEATURES_GP2D_CSV
    elif split == "test":
        batch_paths = config.BatchConfig(config.DEFAULT_TEST_BATCH if batch is None else batch)
        fs, fg, fg2 = (
            batch_paths.features_test_csv,
            batch_paths.features_gp_test_csv,
            batch_paths.features_gp2d_test_csv,
        )
    else:
        raise ValueError("split must be 'train' or 'test'")
    return est.add_derived_features(
        load_cache(fs, "estadisticas")
        .merge(load_cache(fg, "gp1d"), on="object_id")
        .merge(load_cache(fg2, "gp2d"), on="object_id")
    )


def e2gp_matrix(df, feature_columns=None):
    "Return the numerical feature matrix and its ordered column list.\n\nPass the training column order when preparing test data. Object identifiers\nand target labels are excluded; missing numerical values are replaced by zero."
    cols = (
        feature_columns
        if feature_columns is not None
        else [
            c
            for c in df.columns
            if c not in ("object_id", "target") and is_numeric_dtype(df[c].dtype)
        ]
    )
    X = np.nan_to_num(df[cols].to_numpy(dtype=float), nan=0.0)
    return X, cols


def e2gp_al(y, random_state=42):
    "Build the canonical E2-GP estimator: 150 trees and three-fold sigmoid calibration."
    return calibrated_al_classifier(y=y, n_estimators=150, cv=3, random_state=random_state)


def _file_sha256(file_path, tamano_bloque=8 * 1024 * 1024):
    "File sha256."
    import hashlib

    summary_table = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for feature_block_name in iter(lambda: file_handle.read(tamano_bloque), b""):
            summary_table.update(feature_block_name)
    return summary_table.hexdigest()


def save_e2gp_artifact(
    estimator_model,
    feature_columns,
    *,
    n_train,
    ruta_modelo=None,
    ruta_metadatos=None,
    ruta_checksums=None,
):
    "Save the fitted estimator, ordered feature contract, software versions and SHA-256 checksums."
    import hashlib
    import json
    import platform

    import joblib
    import sklearn
    from src.features.cache import REPRESENTATION_VERSION

    import config

    ruta_modelo = config.E2GP_MODEL if ruta_modelo is None else ruta_modelo
    ruta_metadatos = config.E2GP_METADATA if ruta_metadatos is None else ruta_metadatos
    ruta_checksums = config.CHECKSUMS_MODELOS if ruta_checksums is None else ruta_checksums
    ruta_modelo, ruta_metadatos, ruta_checksums = map(
        __import__("pathlib").Path, (ruta_modelo, ruta_metadatos, ruta_checksums)
    )
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = list(feature_columns)
    joblib.dump(estimator_model, ruta_modelo, compress=3)
    hash_modelo = _file_sha256(ruta_modelo)
    esquema = json.dumps(feature_columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    model_parameters = estimator_model.get_params(deep=True)
    object_metadata = {
        "artifact_version": 1,
        "representation_version": REPRESENTATION_VERSION,
        "nombre": "Calibrated E2-GP (150 trees)",
        "productor": "notebooks/04_final_classification.ipynb",
        "n_train": int(n_train),
        "n_features": len(feature_columns),
        "features": feature_columns,
        "feature_schema_sha256": hashlib.sha256(esquema).hexdigest(),
        "classes": [int(c) for c in estimator_model.classes_],
        "especificacion": {
            "estimador": "RandomForestClassifier",
            "n_estimators": int(model_parameters["estimator__n_estimators"]),
            "min_samples_leaf": int(model_parameters["estimator__min_samples_leaf"]),
            "class_weight": model_parameters["estimator__class_weight"],
            "random_state": int(model_parameters["estimator__random_state"]),
            "calibracion": estimator_model.method,
            "cv": int(estimator_model.cv),
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "model_sha256": hash_modelo,
    }
    ruta_metadatos.write_text(
        json.dumps(object_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hash_metadatos = _file_sha256(ruta_metadatos)
    ruta_checksums.write_text(
        f"{hash_modelo}  {ruta_modelo.name}\n" f"{hash_metadatos}  {ruta_metadatos.name}\n",
        encoding="utf-8",
    )
    return object_metadata


def load_e2gp_artifact(*, ruta_modelo=None, ruta_metadatos=None):
    "Load the canonical artifact only after checking representation, integrity and scikit-learn compatibility."
    import json
    from pathlib import Path

    import joblib
    import sklearn

    import config
    from src.features.cache import REPRESENTATION_VERSION

    ruta_modelo = Path(config.E2GP_MODEL if ruta_modelo is None else ruta_modelo)
    ruta_metadatos = Path(config.E2GP_METADATA if ruta_metadatos is None else ruta_metadatos)
    object_metadata = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    if object_metadata.get("representation_version") != REPRESENTATION_VERSION:
        raise RuntimeError(
            "The model representation version does not match the current E2-GP contract."
        )
    actual_value = _file_sha256(ruta_modelo)
    if actual_value != object_metadata["model_sha256"]:
        raise ValueError(f"Checksum mismatch for {ruta_modelo.name}")
    version = object_metadata["software"]["scikit_learn"]
    if version != sklearn.__version__:
        raise RuntimeError(
            f"The artifact requires scikit-learn {version}; current environment: {sklearn.__version__}"
        )
    estimator_model = joblib.load(ruta_modelo)
    if int(estimator_model.n_features_in_) != object_metadata["n_features"]:
        raise ValueError("The estimator and feature contract do not match")
    return estimator_model, object_metadata["features"], object_metadata
