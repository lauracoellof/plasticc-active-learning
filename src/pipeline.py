"""Load or construct batch features and run matched active-learning experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data.cleaning import cleaning_pipeline, synchronize_metadata
from src.features import statistics as est


from src.models import classification
from src.models import active_learning as al


CLASES_TRAIN = (6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95)


def available_batches():
    """Return raw test-batch numbers found under the configured data directory."""
    import re

    ns = []
    for f in config.DIR_RAW.glob("plasticc_test_lightcurves_*.csv.gz"):
        m = re.search("_(\\d+)\\.csv\\.gz$", f.name)
        if m:
            ns.append(int(m.group(1)))
    return sorted(ns)


def select_batch(batch):
    """Construct validated paths for one test batch or the combined pool."""
    return config.BatchConfig(batch)


def prompt_batch(disponibles=None):
    """Ask for a batch number, restricting choices to available raw batches when possible."""
    disp = disponibles if disponibles is not None else available_batches()
    if not disp:
        return int(input("Batch number to process: ").strip())
    print(f"Batches available in data/raw/: {disp}")
    while True:
        r = input(f"Which batch should be processed? {disp}: ").strip()
        if r.isdigit() and int(r) in disp:
            return int(r)
        print(f"  '{r}' is invalid. Choose one of {disp}.")


_MAX_OBJ_CONFIGURADO = object()


def clean_batch(
    lc_path=None,
    meta_path=None,
    *,
    batch_paths=None,
    snr_min=None,
    min_obs=None,
    max_obj=_MAX_OBJ_CONFIGURADO,
    random_state=None,
    verbose=True,
):
    """Clean one raw batch and optionally draw a fixed-size object subsample without replacement.

    Return aligned light-curve and metadata tables. Pass max_obj=None to retain
    the full cleaned batch; omitting it uses the configured subsample size."""
    batch_paths = (
        config.BatchConfig(config.DEFAULT_TEST_BATCH) if batch_paths is None else batch_paths
    )
    lc_path = lc_path or batch_paths.test_lc
    meta_path = meta_path or batch_paths.test_metadata_csv
    snr_min = config.SNR_MIN if snr_min is None else snr_min
    min_obs = config.MIN_OBS if min_obs is None else min_obs
    if max_obj is _MAX_OBJ_CONFIGURADO:
        max_obj = config.TEST_MAX_OBJ
    random_state = config.RANDOM_STATE if random_state is None else random_state

    df = pd.read_csv(lc_path).rename(columns={"detected_bool": "detected"})
    meta_full = pd.read_csv(meta_path)
    metadata_table = meta_full[meta_full["object_id"].isin(df["object_id"].unique())].copy()

    df_clean = cleaning_pipeline(df, snr_min=snr_min, min_obs=min_obs, verbose=verbose)
    meta_clean = synchronize_metadata(metadata_table, df_clean)

    n = meta_clean["object_id"].nunique()
    if max_obj is not None and n > max_obj:

        rng = np.random.default_rng(random_state)
        object_ids = rng.choice(meta_clean["object_id"].values, size=max_obj, replace=False)
        meta_clean = meta_clean[meta_clean["object_id"].isin(object_ids)].copy()
        df_clean = df_clean[df_clean["object_id"].isin(object_ids)].copy()
        if verbose:
            print(f"subsampling: {n:,} -> {meta_clean['object_id'].nunique():,} objects")

    assert set(df_clean["object_id"]) == set(meta_clean["object_id"])
    return df_clean, meta_clean


def features_batch(
    curvas, metadata_table, *, batch_paths=None, usar_cache=True, guardar_cache=False, verbose=True
):
    """Load validated batch caches or extract all three feature families from supplied curves.

    When writing caches, record their schema, source and byte hashes. Derived
    features are appended only after merging the three base feature tables."""
    from pathlib import Path
    from src.features.cache import load_cache, write_cache, validate_cache

    batch_paths = (
        config.BatchConfig(config.DEFAULT_TEST_BATCH) if batch_paths is None else batch_paths
    )
    file_paths = (
        batch_paths.features_test_csv,
        batch_paths.features_gp_test_csv,
        batch_paths.features_gp2d_test_csv,
    )
    cache_valida = False
    if usar_cache and all(Path(r).exists() for r in file_paths):
        try:
            for file_path, feature_family_name in zip(file_paths, ("estadisticas", "gp1d", "gp2d")):
                validate_cache(file_path, feature_family_name, verify_csv=True)
            cache_valida = True
        except Exception as exc:
            if verbose:
                print(f"features: cache discarded ({exc})")
    if cache_valida:
        if verbose:
            print("features: using validated test-feature CSVs")
        df = (
            load_cache(file_paths[0], "estadisticas")
            .merge(load_cache(file_paths[1], "gp1d"), on="object_id")
            .merge(load_cache(file_paths[2], "gp2d"), on="object_id")
        )
    else:
        if verbose:
            print("features: computing from light curves (statistics + GP1D + GP2D)")
        from src.features import gp as cgp
        from src.features import gp2d as cgp2d

        stats = est.build_feature_table(curvas, metadata_table)
        tgp = cgp.build_gp_table(curvas, progress_interval=500 if verbose else 0)
        tgp2 = cgp2d.build_gp2d_table(curvas, progress_interval=500 if verbose else 0)
        df = stats.merge(tgp, on="object_id").merge(tgp2, on="object_id")
        if guardar_cache:
            producer_name = "src.pipeline.features_batch"
            write_cache(
                stats, batch_paths.features_test_csv, "estadisticas", producer_name=producer_name
            )
            write_cache(tgp, batch_paths.features_gp_test_csv, "gp1d", producer_name=producer_name)
            write_cache(
                tgp2, batch_paths.features_gp2d_test_csv, "gp2d", producer_name=producer_name
            )
            if verbose:
                print(f"features cached in {batch_paths.features_test_csv.parent}")

    return est.add_derived_features(df)


def _load_batch(batch_paths, lc_path, meta_path, usar_cache, save_features, random_state, verbose):
    """Reuse cleaned metadata only when all feature caches validate; otherwise clean the raw batch."""
    from pathlib import Path
    from src.features.cache import InvalidFeatureCache, validate_cache

    feats_cached = all(
        Path(r).exists()
        for r in (
            batch_paths.features_test_csv,
            batch_paths.features_gp_test_csv,
            batch_paths.features_gp2d_test_csv,
        )
    )
    cache_valida = False
    if usar_cache and feats_cached:
        try:
            for file_path, feature_family_name in zip(
                (
                    batch_paths.features_test_csv,
                    batch_paths.features_gp_test_csv,
                    batch_paths.features_gp2d_test_csv,
                ),
                ("estadisticas", "gp1d", "gp2d"),
            ):
                validate_cache(file_path, feature_family_name, verify_csv=True)
            cache_valida = True
        except InvalidFeatureCache:
            cache_valida = False
    if cache_valida and Path(batch_paths.test_metadata_clean_csv).exists():
        if verbose:
            print("batch: reusing cleaned metadata without reading raw light curves")
        return None, pd.read_csv(batch_paths.test_metadata_clean_csv)
    curvas, metadata_table = clean_batch(
        lc_path, meta_path, batch_paths=batch_paths, random_state=random_state, verbose=verbose
    )
    if save_features:
        metadata_table.to_csv(batch_paths.test_metadata_clean_csv, index=False)
    return curvas, metadata_table


def _batch_predictions(X_tr, y_tr, X_pool, oids_pool, make_model):
    """Fit a known-class estimator and return aligned predicted labels and maximum probabilities."""
    clf = make_model(y_tr).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_pool)
    pred = pd.DataFrame(
        {
            "object_id": oids_pool.astype(int),
            "prediccion": clf.classes_[proba.argmax(1)],
            "confianza": proba.max(1),
        }
    )
    return pred, clf


def process_batch_al(
    batch=None,
    lc_path=None,
    meta_path=None,
    *,
    make_model=None,
    usar_cache=True,
    save_features=True,
    modes_list=("triage", "random"),
    random_seeds=(0, 1, 2, 3, 4),
    conf_anom=0.30,
    conf_dudoso=0.60,
    orden_dudoso="entropy",
    budget=1000,
    b_iter=100,
    n_iter=30,
    save=False,
    random_state=None,
    verbose=True,
):
    """Prepare one batch, run matched triage baselines and optionally save their trajectories.

    Known classes with insufficient examples for stratification are excluded
    from this experiment. Oracle anomaly labels remain outside classifier training."""
    random_state = config.RANDOM_STATE if random_state is None else random_state
    if make_model is None:
        make_model = lambda y: classification.e2gp_al(y=y, random_state=random_state)

    if batch is None:
        batch = prompt_batch()
    batch_paths = select_batch(batch)
    if verbose:
        print(f"Selected batch: {batch}  ·  {batch_paths.test_lc.name}")

    curvas, metadata_table = _load_batch(
        batch_paths, lc_path, meta_path, usar_cache, save_features, random_state, verbose
    )
    df_batch = features_batch(
        curvas,
        metadata_table,
        batch_paths=batch_paths,
        usar_cache=usar_cache,
        guardar_cache=save_features,
        verbose=verbose,
    )
    comunes = set(df_batch["object_id"]) & set(metadata_table["object_id"])
    df_batch = df_batch[df_batch["object_id"].isin(comunes)].reset_index(drop=True)
    metadata_table = metadata_table[metadata_table["object_id"].isin(comunes)].copy()

    df_tr = classification.features_e2gp("train")
    X_tr, cols = classification.e2gp_matrix(df_tr)
    y_tr = df_tr["target"].to_numpy()
    df_batch = df_batch.merge(metadata_table[["object_id", "true_target"]], on="object_id")
    X_pool, _ = classification.e2gp_matrix(df_batch, feature_columns=cols)
    tt = df_batch["true_target"].to_numpy()
    oids = df_batch["object_id"].to_numpy()
    if verbose:
        print(
            f"train {X_tr.shape} · batch/pool {X_pool.shape} · "
            f"class 99x in the batch: {int((tt >= 990).sum())}"
        )

    known_objects = tt < 990
    vals, cnt = np.unique(tt[known_objects], return_counts=True)
    singleton = set(vals[cnt < 2].tolist())
    if singleton:
        quitar = known_objects & np.isin(tt, list(singleton))
        if verbose:
            print(
                f"Warning: {int(quitar.sum())} known-class object(s) with only one "
                f"example {sorted(singleton)} excluded from triage because stratification is undefined."
            )
        keep = ~quitar
        X_tri, tt_tri = X_pool[keep], tt[keep]
    else:
        X_tri, tt_tri = X_pool, tt

    output_path = str(config.DIR_PROCESSED / f"al_triaje_batch{batch}.csv") if save else None
    historico = al.confidence_triage_multiseed(
        X_tr,
        y_tr,
        X_tri,
        tt_tri,
        make_model,
        modes_list=modes_list,
        random_seeds=random_seeds,
        conf_anom=conf_anom,
        conf_dudoso=conf_dudoso,
        orden_dudoso=orden_dudoso,
        budget=budget,
        b_iter=b_iter,
        n_iter=n_iter,
        output_path=output_path,
        verbose=verbose,
    )

    light_curve = historico.groupby(["modo", "iter"])["MF1"].mean().reset_index()
    filas_res = []
    for m in historico["modo"].unique():
        d = light_curve[light_curve["modo"] == m]
        base, fin = float(d["MF1"].iloc[0]), float(d["MF1"].iloc[-1])
        filas_res.append(
            {
                "modo": m,
                "MF1_sin_AL": round(base, 4),
                "MF1_con_AL": round(fin, 4),
                "ganancia": round(fin - base, 4),
            }
        )
    summary_table = pd.DataFrame(filas_res)

    predicciones, _ = _batch_predictions(X_tr, y_tr, X_pool, oids, make_model)

    if save:
        predicciones.to_csv(batch_paths.pred_test_csv, index=False)
        if verbose:
            print(f"saved: {batch_paths.pred_test_csv.name} · to the_triaje_batch{batch}.csv")

    if verbose:
        print("\nSummary (mean of seeds):")
        print(summary_table.to_string(index=False))

    return {
        "historico": historico,
        "resumen": summary_table,
        "predicciones": predicciones,
        "meta": metadata_table,
        "rutas": batch_paths,
    }
