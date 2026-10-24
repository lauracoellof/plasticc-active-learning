"Analyze representation sensitivity."

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.features import gp
from src.data.cleaning import cleaning_pipeline


def cleaning_sensitivity(df_crudo, metadata):
    p_ref = metadata["target"].value_counts(normalize=True)
    records_list = []
    for snr in (2, 3, 5):
        for min_obs in (3, 5, 10):
            limpio = cleaning_pipeline(df_crudo, snr_min=snr, min_obs=min_obs, verbose=False)
            object_ids = limpio["object_id"].unique()
            p = metadata[metadata["object_id"].isin(object_ids)]["target"].value_counts(
                normalize=True
            )
            p, ref = p.align(p_ref, fill_value=0.0)
            class_counts = limpio.groupby(["object_id", "passband"]).size()
            por_objeto = limpio.groupby("object_id").agg(
                n_obs=("mjd", "size"), n_bandas=("passband", "nunique")
            )
            records_list.append(
                {
                    "snr_min": snr,
                    "min_obs": min_obs,
                    "n_observaciones": len(limpio),
                    "n_objetos": len(object_ids),
                    "retencion_objetos_pct": 100.0
                    * len(object_ids)
                    / metadata["object_id"].nunique(),
                    "distancia_tv_clases": 0.5 * np.abs(p - ref).sum(),
                    "bandas_validas_gp1d": int((class_counts >= gp.MIN_OBS_BANDA).sum()),
                    "objetos_validos_gp1d": int(
                        class_counts[class_counts >= gp.MIN_OBS_BANDA]
                        .index.get_level_values(0)
                        .nunique()
                    ),
                    "objetos_validos_gp2d": int(
                        ((por_objeto["n_obs"] >= 8) & (por_objeto["n_bandas"] >= 3)).sum()
                    ),
                }
            )
    return pd.DataFrame(records_list)


def _evaluate_gp(light_curve, band_index, ls_min):
    previo = gp.LS_MIN
    gp.LS_MIN = ls_min
    try:
        fit_result = gp.fit_object_gp(light_curve, band_index)
        if fit_result is None:
            return None
        f = gp._band_features(fit_result)
        return {
            "ls_min": ls_min,
            "lenscale": f["gp_lenscale"],
            "chi2": f["gp_chi2"],
            "resid_std": f["gp_resid_std"],
            "loglik": f["gp_loglik"],
            "en_cota": float(f["gp_lenscale"] <= 1.01 * ls_min),
        }
    finally:
        gp.LS_MIN = previo


def gp1d_sensitivity(df_limpio, metadata):
    rng = np.random.default_rng(config.RANDOM_STATE)
    object_ids = []
    for _, sub in metadata.groupby("target"):
        candidatos = sub["object_id"].to_numpy()
        object_ids.extend(rng.choice(candidatos, size=min(15, len(candidatos)), replace=False))
    muestra = df_limpio[df_limpio["object_id"].isin(object_ids)]
    tareas = [
        (light_curve, int(band_index), ls)
        for (_, band_index), light_curve in muestra.groupby(["object_id", "passband"], sort=False)
        if len(light_curve) >= gp.MIN_OBS_BANDA
        for ls in (1.0, 5.0, 10.0, 20.0)
    ]
    records_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(_evaluate_gp)(light_curve, band_index, ls) for light_curve, band_index, ls in tareas
    )
    res = pd.DataFrame([f for f in records_list if f is not None])
    res["abs_log_chi2"] = np.abs(np.log(np.clip(res["chi2"], 1e-12, None)))
    return res.groupby("ls_min", as_index=False).agg(
        n_ajustes=("chi2", "size"),
        lenscale_mediana=("lenscale", "median"),
        fraccion_en_cota=("en_cota", "mean"),
        chi2_mediana=("chi2", "median"),
        resid_std_mediana=("resid_std", "median"),
        abs_log_chi2_mediana=("abs_log_chi2", "median"),
        loglik_mediana=("loglik", "median"),
    )


def main():
    crudo = pd.read_csv(config.TRAIN_CSV)
    metadata_table = pd.read_csv(config.METADATA_CSV, usecols=["object_id", "target"])
    limpio = pd.read_csv(config.TRAIN_CLEAN_CSV)
    meta_limpia = pd.read_csv(config.METADATA_CLEAN_CSV, usecols=["object_id", "target"])
    s_limpieza = cleaning_sensitivity(crudo, metadata_table)
    s_gp = gp1d_sensitivity(limpio, meta_limpia)
    s_limpieza.to_csv(config.DIR_PROCESSED / "sensibilidad_limpieza.csv", index=False)
    s_gp.to_csv(config.DIR_PROCESSED / "sensibilidad_gp1d_ls.csv", index=False)
    print(s_limpieza.to_string(index=False))
    print(s_gp.to_string(index=False))


if __name__ == "__main__":
    main()
