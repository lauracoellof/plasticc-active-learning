"""Regenerate the complete ten-seed combined-pool allocation sweep.

Requires the combined-pool and training feature caches. Writes only the final
trajectory table; no historical five-seed or partial-sweep input is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import pipeline
from src.models import active_learning as al
from src.models import anomalies as an
from src.models import classification

BATCH_COMB = 99
FRACS = tuple(round(x, 1) for x in np.arange(0.1, 0.9, 0.1))
SEEDS = tuple(range(10))
OUTPUT = config.DIR_PROCESSED / "dos_canales_batch99_fino_10s.csv"


def run(seeds=SEEDS, output_path=OUTPUT):
    """Fit each allocation policy on matched per-seed evaluation splits.

    This is a full AL experiment and may take hours. Existing output is replaced
    by the multiseed driver; preserve it before comparing a new configuration.
    """
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct random seeds")
    tr = classification.features_e2gp("train")
    X_tr, feature_columns = classification.e2gp_matrix(tr)
    y_tr = tr["target"].to_numpy()

    batch_paths = pipeline.select_batch(BATCH_COMB)
    te = classification.features_e2gp("test", batch=BATCH_COMB)
    metadata_table = pd.read_csv(batch_paths.test_metadata_clean_csv)
    te = te.merge(metadata_table[["object_id", "true_target"]], on="object_id")
    y_pool = te["true_target"].to_numpy()
    X_pool, _ = classification.e2gp_matrix(te, feature_columns)

    modelo_inicial = classification.e2gp_al(y=y_tr, random_state=config.RANDOM_STATE).fit(
        X_tr, y_tr
    )
    gp_pool = te[["object_id"]].merge(
        pd.read_csv(batch_paths.features_gp_test_csv), on="object_id", how="left"
    )
    stats_pool = te[["object_id"]].merge(
        pd.read_csv(batch_paths.features_test_csv), on="object_id", how="left"
    )
    puntuaciones = np.column_stack(
        [
            an.confidence_novelty_score(modelo_inicial.predict_proba(X_pool)),
            an.score_isolation_forest(X_pool),
            an.gp_residual_score(gp_pool),
            an.amplitude_score(stats_pool),
        ]
    )
    pool_descriptors = puntuaciones[:, 1:]

    known_objects = y_pool < 990
    class_codes, cuentas = np.unique(y_pool[known_objects], return_counts=True)
    clases_solitarias = set(class_codes[cuentas < 2].tolist())
    retain_count = ~(known_objects & np.isin(y_pool, list(clases_solitarias)))
    X_pool = X_pool[retain_count]
    y_pool = y_pool[retain_count]
    pool_descriptors = pool_descriptors[retain_count]

    def make_model(y_etiquetado):
        return classification.e2gp_al(y=y_etiquetado, random_state=config.RANDOM_STATE)

    print(f"Allocation sweep: batch={BATCH_COMB}, fractions={FRACS}, seeds={seeds}", flush=True)
    return al.two_channel_multiseed(
        X_tr,
        y_tr,
        X_pool,
        y_pool,
        pool_descriptors,
        make_model,
        fracs=FRACS,
        random_seeds=seeds,
        conf_dudoso=0.60,
        reg_C=0.1,
        budget=1000,
        b_iter=100,
        n_iter=30,
        output_path=str(output_path),
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    results = run(seeds=args.seeds, output_path=args.output)
    print(f"Saved {len(results):,} rows to {args.output}")


if __name__ == "__main__":
    main()
