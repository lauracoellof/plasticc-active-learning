"Regenerate e2gp oof calibration."

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.models import classification
from src.models import evaluation as ev


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Optional directory in which to preserve the previous cache.",
    )
    args = parser.parse_args()

    cache = config.DIR_PROCESSED / "calibracion_e2gp_oof_150_cv3.npz"
    manifest = cache.with_suffix(".manifest.json")
    if cache.exists() and args.backup_dir is not None:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache, args.backup_dir / f"{cache.stem}.pre_e2gp2.npz")

    train = classification.features_e2gp("train")
    X_train, columns = classification.e2gp_matrix(train)
    y_train = train["target"].to_numpy()
    classes = np.sort(np.unique(y_train))
    seed = config.RANDOM_STATE
    outer_cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    uncalibrated = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    calibrated = classification.e2gp_al(y=y_train, random_state=seed)

    print(
        f"E2-GP train: {len(y_train):,} objects, "
        f"{len(columns)} variables, {len(classes)} classes"
    )
    print("Computing probabilidades OOF without calibrar (5 folds)...", flush=True)
    proba_uncalibrated = cross_val_predict(
        uncalibrated,
        X_train,
        y_train,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )
    print("Computing probabilidades OOF calibradas (5 x 3 folds)...", flush=True)
    proba_calibrated = cross_val_predict(
        calibrated,
        X_train,
        y_train,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )

    atomic_savez(
        cache,
        proba_sin=proba_uncalibrated,
        proba_cal=proba_calibrated,
        clases=classes,
    )

    pred_calibrated = classes[np.argmax(proba_calibrated, axis=1)]
    ece_uncalibrated, _ = ev.confidence_ece(y_train, proba_uncalibrated, classes)
    ece_calibrated, _ = ev.confidence_ece(y_train, proba_calibrated, classes)
    metrics = {
        "accuracy_oof": float(accuracy_score(y_train, pred_calibrated)),
        "macro_f1_oof": float(f1_score(y_train, pred_calibrated, average="macro", zero_division=0)),
        "weighted_f1_oof": float(
            f1_score(y_train, pred_calibrated, average="weighted", zero_division=0)
        ),
        "ece_uncalibrated": float(ece_uncalibrated),
        "ece_calibrated": float(ece_calibrated),
        "brier_multiclass_uncalibrated": float(
            ev.multiclass_brier_score(y_train, proba_uncalibrated, classes)
        ),
        "brier_multiclass_calibrated": float(
            ev.multiclass_brier_score(y_train, proba_calibrated, classes)
        ),
    }
    source_paths = {
        "features_statistics": Path(config.FEATURES_CSV),
        "features_gp1d": Path(config.FEATURES_GP_CSV),
        "features_gp2d": Path(config.FEATURES_GP2D_CSV),
        "classifier_code": ROOT / "src" / "models" / "classification.py",
        "statistics_code": ROOT / "src" / "features" / "statistics.py",
    }
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "outer_cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "estimator": "RandomForestClassifier",
            "n_estimators": 150,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "calibration_method": "sigmoid",
            "calibration_cv": 3,
            "random_state": int(seed),
        },
        "data": {
            "n_objects": int(len(y_train)),
            "n_features": int(len(columns)),
            "classes": classes.tolist(),
            "feature_columns": list(columns),
        },
        "source_sha256": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "output": {
            "path": str(cache.resolve()),
            "sha256": sha256(cache),
        },
        "metrics": metrics,
    }
    manifest.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Cache: {cache.resolve()}")
    print(f"Manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
