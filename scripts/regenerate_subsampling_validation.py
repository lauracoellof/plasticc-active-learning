"Regenerate subsampling validation."

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import sklearn

import config
from src.features.cache import load_cache
from src.models import anomalies, classification
from src.models.subsampling import evaluate_subsampling


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4, choices=range(2, 12))
    parser.add_argument("--resamples", "--remuestreos", dest="resamples", type=int, default=200)
    args = parser.parse_args()
    batch_paths = config.BatchConfig(args.batch)
    full_path = config.DIR_PROCESSED / f"features_e2gp_test_b{args.batch}_full.csv"
    model, cols, model_meta = classification.load_e2gp_artifact()
    full = load_cache(full_path, "e2gp")
    sub_features = classification.features_e2gp("test", batch=args.batch)
    if full.object_id.duplicated().any() or sub_features.object_id.duplicated().any():
        raise ValueError("Duplicate object identifiers in the feature matrices")
    full = full.set_index("object_id", drop=False)
    sub_features = sub_features.set_index("object_id", drop=False)
    if not sub_features.index.isin(full.index).all():
        raise ValueError("The subsample is not contained in the full batch")
    before = full.loc[sub_features.index, cols].to_numpy(float)
    canonical = sub_features[cols].to_numpy(float)
    changed = ~np.isclose(before, canonical, rtol=1e-9, atol=1e-9, equal_nan=True)

    full.loc[sub_features.index, cols] = canonical
    if "true_target" not in full:
        truth = pd.read_csv(
            config.TEST_METADATA_CSV, usecols=["object_id", "true_target"]
        ).set_index("object_id")
        full["true_target"] = truth.reindex(full.index)["true_target"]
    if full.true_target.isna().any():
        raise ValueError("Missing ground-truth labels in the full batch")
    X, _ = classification.e2gp_matrix(full, cols)

    for calibrated in model.calibrated_classifiers_:
        calibrated.estimator.set_params(n_jobs=1)
    print(f"Predicting {len(full):,} objects with the canonical model...", flush=True)
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    sub = full.index.isin(sub_features.index)
    predictions = pd.read_csv(batch_paths.pred_test_csv).set_index("object_id")
    predicted_sub = pd.Series(conf, index=full.index).reindex(predictions.index)
    if not np.allclose(predicted_sub, predictions.proba_max, rtol=1e-8, atol=1e-8):
        raise ValueError(
            "The model does not reproduce the canonical subsample probabilities; no results were written"
        )
    tests, summary, samples = evaluate_subsampling(
        conf,
        full.true_target.to_numpy(),
        sub,
        isolation=anomalies.score_isolation_forest(X),
        residuos=anomalies.gp_residual_score(full),
        n_remuestreos=args.resamples,
        seed_value=config.RANDOM_STATE,
    )
    prefix = (
        "validacion_submuestreo" if args.batch == 4 else f"validacion_submuestreo_b{args.batch}"
    )
    outputs = {}
    for suffix, table in [("tests", tests), ("remuestreo", summary), ("realizaciones", samples)]:
        path = config.DIR_PROCESSED / f"{prefix}_{suffix}.csv"
        table.to_csv(path, index=False)
        outputs[path.name] = sha256(path)
    inputs = [
        full_path,
        batch_paths.features_test_csv,
        batch_paths.features_gp_test_csv,
        batch_paths.features_gp2d_test_csv,
        batch_paths.pred_test_csv,
        config.E2GP_MODEL,
        config.E2GP_METADATA,
        config.TEST_METADATA_CSV,
    ]
    provenance = {
        "protocol_version": 1,
        "batch": args.batch,
        "seed": config.RANDOM_STATE,
        "n_remuestreos": args.resamples,
        "n_full": len(full),
        "n_sub": int(sub.sum()),
        "continuous_summary": "median",
        "model_specification": model_meta["especificacion"],
        "aligned_feature_rows": int(changed.any(axis=1).sum()),
        "aligned_feature_columns": [c for c, m in zip(cols, changed.any(axis=0)) if m],
        "alignment": "canonical subsample features replace shared rows in memory before scoring",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in inputs},
        "code_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in [
                Path(__file__),
                ROOT / "src/models/subsampling.py",
                ROOT / "src/models/anomalies.py",
            ]
        },
        "outputs_sha256": outputs,
    }
    (config.DIR_PROCESSED / f"{prefix}.manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(tests.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
