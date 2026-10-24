"Run isolated block ablation."

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import config
from src.features import statistics as est
from src.models.classification import calibrated_al_classifier


SEEDS = tuple(range(10))
TEST_SIZE = 0.20
SEMILLA_ESTIMADOR = 42

RUTA_RESULTADOS = config.DIR_PROCESSED / "ablacion_bloques_aislados_por_semilla.csv"
RUTA_RESUMEN = config.DIR_PROCESSED / "ablacion_bloques_aislados_resumen.csv"
RUTA_PARTICIONES = config.DIR_PROCESSED / "ablacion_bloques_aislados_particiones.csv"
RUTA_METADATOS = config.DIR_PROCESSED / "ablacion_bloques_aislados.meta.json"


def _sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for feature_block_name in iter(lambda: f.read(1024 * 1024), b""):
            h.update(feature_block_name)
    return h.hexdigest()


def _load_blocks():
    stats = pd.read_csv(config.FEATURES_CSV)
    gp1 = pd.read_csv(config.FEATURES_GP_CSV)
    gp2 = pd.read_csv(config.FEATURES_GP2D_CSV)

    for item_name, df in (("statistics", stats), ("GP1D", gp1), ("GP2D", gp2)):
        if "object_id" not in df:
            raise ValueError(f"Missing object_id in the table of {item_name}")
        if df["object_id"].duplicated().any():
            raise ValueError(f"The table of {item_name} contains duplicate object identifiers")

    ids_comunes = set(stats["object_id"]) & set(gp1["object_id"]) & set(gp2["object_id"])
    if not ids_comunes:
        raise ValueError("The tres familias of features not comparten objects")

    base = (
        stats[stats["object_id"].isin(ids_comunes)].sort_values("object_id").reset_index(drop=True)
    )
    gp1_alineado = (
        gp1[gp1["object_id"].isin(ids_comunes)].sort_values("object_id").reset_index(drop=True)
    )
    gp2_alineado = (
        gp2[gp2["object_id"].isin(ids_comunes)].sort_values("object_id").reset_index(drop=True)
    )

    ids_ref = base["object_id"].to_numpy()
    y_ref = base["target"].to_numpy()
    if not np.array_equal(gp1_alineado["object_id"].to_numpy(), ids_ref):
        raise AssertionError("GP1D not preserves the orden of objects")
    if not np.array_equal(gp2_alineado["object_id"].to_numpy(), ids_ref):
        raise AssertionError("GP2D not preserves the orden of objects")

    stats_gp1 = base.merge(gp1_alineado, on="object_id", how="inner", validate="one_to_one")
    stats_gp1_der = est.add_derived_features(stats_gp1.copy())
    columnas_derivadas = [c for c in stats_gp1_der.columns if c not in stats_gp1.columns]

    marcos_y_columnas = {
        "Statistics": (base, [c for c in base.columns if c not in ("object_id", "target")]),
        "GP1D": (gp1_alineado, [c for c in gp1_alineado.columns if c != "object_id"]),
        "Derivadas": (stats_gp1_der, columnas_derivadas),
        "GP2D": (gp2_alineado, [c for c in gp2_alineado.columns if c != "object_id"]),
    }

    matrices = {}
    for item_name, (df, feature_columns) in marcos_y_columnas.items():
        X = df[feature_columns].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError(f"The bloque {item_name} contains values not finite")
        matrices[item_name] = (X, feature_columns)

    class_counts = {
        item_name: len(feature_columns) for item_name, (_, feature_columns) in matrices.items()
    }
    expected_entries = {"Statistics": 91, "GP1D": 136, "Derivadas": 24, "GP2D": 53}
    if class_counts != expected_entries:
        raise AssertionError(f"Schema of bloques inesperado: {class_counts}")
    return ids_ref, y_ref, matrices


def run():
    object_ids, y, matrices = _load_blocks()
    records_list, split_records = [], []
    indices = np.arange(len(y))

    for seed_value in SEEDS:
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_SIZE, random_state=seed_value, stratify=y
        )
        split = np.full(len(y), "train", dtype=object)
        split[idx_test] = "test"
        split_records.extend(
            {"semilla": seed_value, "object_id": int(obj), "particion": object_group}
            for obj, object_group in zip(object_ids, split)
        )

        for feature_block_name, (X, feature_columns) in matrices.items():
            estimator_model = calibrated_al_classifier(
                y=y[idx_train],
                n_estimators=150,
                min_samples_leaf=3,
                method_name="sigmoid",
                cv=3,
                random_state=SEMILLA_ESTIMADOR,
                n_jobs=-1,
            )
            estimator_model.fit(X[idx_train], y[idx_train])
            pred = estimator_model.predict(X[idx_test])
            records_list.append(
                {
                    "semilla": seed_value,
                    "bloque": feature_block_name,
                    "n_features": len(feature_columns),
                    "n_train": len(idx_train),
                    "n_test": len(idx_test),
                    "MF1": f1_score(y[idx_test], pred, average="macro", zero_division=0),
                    "WF1": f1_score(y[idx_test], pred, average="weighted", zero_division=0),
                    "accuracy": accuracy_score(y[idx_test], pred),
                }
            )

    result_records = pd.DataFrame(records_list)
    summary_table = (
        result_records.groupby(["bloque", "n_features"], sort=False)[["MF1", "WF1", "accuracy"]]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    summary_table.columns = ["_".join(c).rstrip("_") for c in summary_table.columns.to_flat_index()]

    result_records.to_csv(RUTA_RESULTADOS, index=False)
    summary_table.to_csv(RUTA_RESUMEN, index=False)
    pd.DataFrame(split_records).to_csv(RUTA_PARTICIONES, index=False)

    object_metadata = {
        "objetivo": "capacidad predictiva of each bloque of E2-GP utilizado by separado",
        "interpretacion": (
            "diagnostico of suficiencia predictiva; the contribucion incremental  estima "
            "in ablacion_causal_representacion"
        ),
        "n_objetos_comunes": int(len(object_ids)),
        "semillas_particion": list(SEEDS),
        "test_size": TEST_SIZE,
        "clasificador": {
            "tipo": "RandomForestClassifier calibrated by Platt",
            "n_estimators": 150,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "calibracion": "sigmoid",
            "cv_calibracion": 3,
            "random_state": SEMILLA_ESTIMADOR,
        },
        "entradas_sha256": {
            str(config.FEATURES_CSV.name): _sha256(config.FEATURES_CSV),
            str(config.FEATURES_GP_CSV.name): _sha256(config.FEATURES_GP_CSV),
            str(config.FEATURES_GP2D_CSV.name): _sha256(config.FEATURES_GP2D_CSV),
        },
    }
    RUTA_METADATOS.write_text(
        json.dumps(object_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_records, summary_table


if __name__ == "__main__":
    _, summary_table = run()
    print("\nRendimiento of the bloques aislados")
    print(summary_table.to_string(index=False))
