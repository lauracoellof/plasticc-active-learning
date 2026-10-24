"Run incremental representation ablation."

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import config
from src.features import statistics as est
from src.models.classification import calibrated_al_classifier


SEEDS = tuple(range(10))
TEST_SIZE = 0.20
SEMILLA_ESTIMADOR = 42
N_BOOT = 20_000

RUTA_RESULTADOS = config.DIR_PROCESSED / "ablacion_causal_representacion_por_semilla.csv"
RUTA_RESUMEN = config.DIR_PROCESSED / "ablacion_causal_representacion_resumen.csv"
RUTA_CONTRASTES = config.DIR_PROCESSED / "ablacion_causal_representacion_contrastes.csv"
RUTA_PARTICIONES = config.DIR_PROCESSED / "ablacion_causal_representacion_particiones.csv"
RUTA_METADATOS = config.DIR_PROCESSED / "ablacion_causal_representacion.meta.json"


def _sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for feature_block_name in iter(lambda: f.read(1024 * 1024), b""):
            h.update(feature_block_name)
    return h.hexdigest()


def _load_variants():
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
    stats_gp1 = base.merge(gp1, on="object_id", how="inner", validate="one_to_one")
    stats_gp1_der = est.add_derived_features(stats_gp1.copy())
    completa = stats_gp1_der.merge(gp2, on="object_id", how="inner", validate="one_to_one")

    marcos = {
        "Statistics": base,
        "Statistics + GP1D": stats_gp1,
        "Statistics + GP1D + derived features": stats_gp1_der,
        "Full E2-GP": completa,
    }
    ids_ref = base["object_id"].to_numpy()
    y_ref = base["target"].to_numpy()
    matrices = {}
    for item_name, df in marcos.items():
        if not np.array_equal(df["object_id"].to_numpy(), ids_ref):
            raise AssertionError(
                f"The variante {item_name} does not preserve the same object order"
            )
        if not np.array_equal(df["target"].to_numpy(), y_ref):
            raise AssertionError(f"The variante {item_name} does not preserve the same labels")
        feature_columns = [c for c in df.columns if c not in ("object_id", "target")]
        X = df[feature_columns].to_numpy()
        if not np.isfinite(X).all():
            raise ValueError(f"The variante {item_name} contains values not finite")
        matrices[item_name] = (X, feature_columns)

    class_counts = {
        item_name: len(feature_columns) for item_name, (_, feature_columns) in matrices.items()
    }
    expected_entries = {
        "Statistics": 91,
        "Statistics + GP1D": 227,
        "Statistics + GP1D + derived features": 251,
        "Full E2-GP": 304,
    }
    if class_counts != expected_entries:
        raise AssertionError(f"Schema of features inesperado: {class_counts}")
    return ids_ref, y_ref, matrices


def _holm(p_valores):
    p = np.asarray(p_valores, dtype=float)
    sort_order = np.argsort(p)
    ajustados_orden = np.minimum(
        1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[sort_order])
    )
    ajustados = np.empty_like(ajustados_orden)
    ajustados[sort_order] = ajustados_orden
    return ajustados


def _paired_tests(result_records: pd.DataFrame) -> pd.DataFrame:
    pares = [
        ("Statistics + GP1D", "Statistics", "Add GP1D"),
        ("Statistics + GP1D + derived features", "Statistics + GP1D", "Add derived features"),
        ("Full E2-GP", "Statistics + GP1D + derived features", "Add GP2D"),
    ]
    rng = np.random.default_rng(SEMILLA_ESTIMADOR)
    records_list = []
    for propuesta, reference_results, contraste in pares:
        a = (
            result_records[result_records["variante"] == propuesta]
            .sort_values("semilla")["MF1"]
            .to_numpy()
        )
        b = (
            result_records[result_records["variante"] == reference_results]
            .sort_values("semilla")["MF1"]
            .to_numpy()
        )
        d = a - b
        sample_records = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
        no_cero = d[d != 0]
        if len(no_cero):
            signed_ranks = rankdata(np.abs(no_cero))
            rb = (
                signed_ranks[no_cero > 0].sum() - signed_ranks[no_cero < 0].sum()
            ) / signed_ranks.sum()
            method_name = "exact" if len(no_cero) == len(d) else "approx"
            p = wilcoxon(d, alternative="two-sided", method=method_name).pvalue
        else:
            rb, p = 0.0, 1.0
        records_list.append(
            {
                "contraste": contraste,
                "referencia": reference_results,
                "propuesta": propuesta,
                "n": len(d),
                "diferencia_media_MF1": d.mean(),
                "ic95_inf": np.quantile(sample_records, 0.025),
                "ic95_sup": np.quantile(sample_records, 0.975),
                "victorias": int((d > 0).sum()),
                "rango_biserial": float(rb),
                "p_wilcoxon": float(p),
            }
        )
    result_table = pd.DataFrame(records_list)
    result_table["p_holm"] = _holm(result_table["p_wilcoxon"])
    return result_table


def run():
    object_ids, y, matrices = _load_variants()
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

        for variant_name, (X, feature_columns) in matrices.items():
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
                    "variante": variant_name,
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
        result_records.groupby(["variante", "n_features"], sort=False)[["MF1", "WF1", "accuracy"]]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    summary_table.columns = ["_".join(c).rstrip("_") for c in summary_table.columns.to_flat_index()]
    contrastes = _paired_tests(result_records)

    result_records.to_csv(RUTA_RESULTADOS, index=False)
    summary_table.to_csv(RUTA_RESUMEN, index=False)
    contrastes.to_csv(RUTA_CONTRASTES, index=False)
    pd.DataFrame(split_records).to_csv(RUTA_PARTICIONES, index=False)

    object_metadata = {
        "objetivo": "efecto incremental of GP1D, transformaciones derived features and GP2D",
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
        "n_boot": N_BOOT,
        "entradas_sha256": {
            str(config.FEATURES_CSV.name): _sha256(config.FEATURES_CSV),
            str(config.FEATURES_GP_CSV.name): _sha256(config.FEATURES_GP_CSV),
            str(config.FEATURES_GP2D_CSV.name): _sha256(config.FEATURES_GP2D_CSV),
        },
    }
    RUTA_METADATOS.write_text(
        json.dumps(object_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_records, summary_table, contrastes


if __name__ == "__main__":
    result_records, summary_table, contrastes = run()
    print("\nSummary by variante")
    print(summary_table.to_string(index=False))
    print("\nTests incrementales of macro-F1")
    print(contrastes.to_string(index=False))
