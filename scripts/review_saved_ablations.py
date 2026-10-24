"""Check saved ablation summaries against feature inputs, seed rows and partitions.

This verifies the saved experiment's internal consistency. It does not refit
forests or establish that saved per-seed predictions are reproducible.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def review(prefix, group_column):
    root = config.DIR_PROCESSED
    metadata = json.loads((root / f"{prefix}.meta.json").read_text(encoding="utf-8"))
    for name, expected in metadata["entradas_sha256"].items():
        path = root / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or modified ablation feature input: {name}")
    frame = pd.read_csv(root / f"{prefix}_por_semilla.csv")
    summary = pd.read_csv(root / f"{prefix}_resumen.csv")
    partitions = pd.read_csv(root / f"{prefix}_particiones.csv")
    seeds = set(metadata["semillas_particion"])
    if frame.duplicated([group_column, "semilla"]).any():
        raise ValueError("Duplicate per-seed experiment rows")
    if any(set(g.semilla) != seeds for _, g in frame.groupby(group_column)):
        raise ValueError("Incomplete ablation seed coverage")
    stats = pd.read_csv(config.FEATURES_CSV)
    ids = set(stats.object_id)
    for path in [config.FEATURES_GP_CSV, config.FEATURES_GP2D_CSV]:
        ids &= set(pd.read_csv(path, usecols=["object_id"]).object_id)
    base = stats[stats.object_id.isin(ids)].sort_values("object_id").reset_index(drop=True)
    if len(base) != metadata["n_objetos_comunes"]:
        raise ValueError("Changed ablation object intersection")
    if partitions.duplicated(["semilla", "object_id"]).any() or set(partitions.semilla) != seeds:
        raise ValueError("Duplicate or incomplete saved partitions")
    for seed in sorted(seeds):
        _, test = train_test_split(
            np.arange(len(base)),
            test_size=metadata["test_size"],
            random_state=seed,
            stratify=base.target,
        )
        expected = pd.Series("train", index=base.object_id)
        expected.loc[base.iloc[test].object_id] = "test"
        observed = partitions[partitions.semilla.eq(seed)].set_index("object_id").particion
        if set(observed.index) != set(expected.index) or not observed.reindex(
            expected.index
        ).equals(expected):
            raise ValueError(f"Saved partition does not reproduce for seed {seed}")
        rows = frame[frame.semilla.eq(seed)]
        if not rows.n_test.eq(len(test)).all() or not rows.n_train.eq(len(base) - len(test)).all():
            raise ValueError("Saved experiment sample counts disagree with partitions")
    recomputed = (
        frame.groupby([group_column, "n_features"], sort=False)[["MF1", "WF1", "accuracy"]]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    recomputed.columns = ["_".join(c).rstrip("_") for c in recomputed.columns.to_flat_index()]
    keys = [group_column, "n_features"]
    pd.testing.assert_frame_equal(
        recomputed.sort_values(keys).reset_index(drop=True),
        summary.sort_values(keys).reset_index(drop=True),
        check_exact=False,
        rtol=1e-11,
        atol=1e-12,
    )
    print(
        "Verified feature hashes, stratified partitions, seed coverage and saved summaries; no model refitted."
    )
    return frame, summary
