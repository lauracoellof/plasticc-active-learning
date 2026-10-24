"""Per-band statistics, cross-band summaries and derived photometric proxies."""

import numpy as np
import pandas as pd

PASSBAND_MAP = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}


BAND_WAVELENGTHS = np.array([354, 477, 623, 762, 913, 1002], dtype=float)

FEATURE_NAMES = [
    "mean",
    "std",
    "median",
    "min",
    "max",
    "count",
    "skew",
    "kurt",
    "q25",
    "q75",
    "iqr",
    "cv",
]

NEW_FEATURES = [
    "spectral_centroid",
    "red_fraction",
    "blue_fraction",
    "spectral_asym",
    "spectral_slope_yg",
    "color_zy",
    "peak_band_idx",
    "peak_std_band",
    "iqr_ratio_ry",
    "n_active_bands",
    "cv_band_std",
    "mean_skew",
    "baseline_days",
    "t_centroid_norm",
    "rise_frac",
]

COLS_META = ["hostgal_photoz", "hostgal_photoz_err", "mwebv", "distmod"]
FLUX_ZEROPOINT = 27.5


def _group_statistics(object_group):
    """Compute the twelve historical flux statistics for one object-band group."""
    f = object_group["flux"]
    q25, q75 = f.quantile(0.25), f.quantile(0.75)
    mean_v = f.mean()
    std_v = f.std(ddof=1) if len(f) > 1 else 0.0
    return pd.Series(
        {
            "mean": mean_v,
            "std": std_v,
            "median": f.median(),
            "min": f.min(),
            "max": f.max(),
            "count": len(f),
            "skew": f.skew(),
            "kurt": f.kurt(),
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "cv": std_v / mean_v if mean_v != 0 else 0.0,
        }
    )


def build_statistics(df):
    """Pivot per-band statistics into a fixed six-band schema, filling unsupported entries with zero."""
    stats = (
        df.groupby(["object_id", "passband"])
        .apply(_group_statistics, include_groups=False)
        .reset_index()
    )
    ancha = stats.pivot_table(index="object_id", columns="passband", values=FEATURE_NAMES)
    ancha.columns = [f"{feat}_{PASSBAND_MAP[int(pb)]}" for feat, pb in ancha.columns]
    cols = [f"{feat}_{b}" for b in PASSBAND_MAP.values() for feat in FEATURE_NAMES]

    ancha = ancha.reindex(columns=cols, fill_value=0.0).fillna(0.0)
    return ancha.reset_index()


def _group_cross_band_features(grp):
    """Compute cross-band flux, variability and temporal summaries for one object."""
    band_means = grp.groupby("passband")["flux"].mean().reindex(range(6), fill_value=0.0)
    arr = band_means.values.clip(0)
    total = arr.sum() + 1e-6

    spectral_centroid = (arr * BAND_WAVELENGTHS).sum() / total
    red_fraction = arr[3:].sum() / total
    blue_fraction = arr[:2].sum() / total
    spectral_asym = (arr[5] - arr[0]) / (arr[5] + arr[0] + 1e-6)
    spectral_slope_yg = arr[5] / (arr[1] + 1e-6)
    color_zy = arr[4] / (arr[5] + 1e-6)
    peak_band_idx = float(np.argmax(arr))

    band_stds = grp.groupby("passband")["flux"].std().reindex(range(6), fill_value=0.0)
    peak_std_band = float(np.argmax(band_stds.values))

    def band_iqr(b):
        v = grp[grp["passband"] == b]["flux"]
        return float(v.quantile(0.75) - v.quantile(0.25)) if len(v) >= 2 else 0.0

    iqr_ratio_ry = band_iqr(2) / (band_iqr(5) + 1e-6)

    n_active_bands = float((arr > 0).sum())

    cv_list = []
    for b in range(6):
        v = grp[grp["passband"] == b]["flux"]
        if len(v) >= 2 and abs(v.mean()) > 1e-6:
            cv_list.append(v.std() / abs(v.mean()))
    cv_band_std = float(np.std(cv_list)) if cv_list else 0.0

    mean_skew = float(
        grp.groupby("passband")["flux"].skew().reindex(range(6), fill_value=0.0).mean()
    )

    grp_s = grp.sort_values("mjd")
    t = grp_s["mjd"].values
    f_vals = grp_s["flux"].values
    baseline = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    f_pos = f_vals.clip(0)
    t_c = (f_pos * t).sum() / (f_pos.sum() + 1e-6)
    t_centroid_norm = (t_c - t[0]) / (baseline + 1e-6)
    peak_idx = int(np.argmax(f_vals))
    rise_frac = (t[peak_idx] - t[0]) / (baseline + 1e-6)

    return pd.Series(
        {
            "spectral_centroid": spectral_centroid,
            "red_fraction": red_fraction,
            "blue_fraction": blue_fraction,
            "spectral_asym": spectral_asym,
            "spectral_slope_yg": spectral_slope_yg,
            "color_zy": color_zy,
            "peak_band_idx": peak_band_idx,
            "peak_std_band": peak_std_band,
            "iqr_ratio_ry": iqr_ratio_ry,
            "n_active_bands": n_active_bands,
            "cv_band_std": cv_band_std,
            "mean_skew": mean_skew,
            "baseline_days": baseline,
            "t_centroid_norm": t_centroid_norm,
            "rise_frac": rise_frac,
        }
    )


def build_cross_band_features(df):
    """Return the cross-band descriptor table keyed by object_id."""
    return (
        df.groupby("object_id")
        .apply(_group_cross_band_features, include_groups=False)
        .reset_index()
    )


def build_feature_table(df, metadata):
    """Join per-band statistics, cross-band descriptors, host metadata and training labels."""
    stats = build_statistics(df)
    new_indices = build_cross_band_features(df)
    result_table = stats.merge(new_indices, on="object_id", how="left")
    result_table = result_table.merge(
        metadata[["object_id"] + COLS_META + ["target"]], on="object_id"
    )

    result_table["distmod"] = result_table["distmod"].fillna(0)

    feat_cols = [c for c in result_table.columns if c not in ("object_id", "target")]
    result_table[feat_cols] = result_table[feat_cols].fillna(0)
    return result_table


def add_derived_features(result_table, band_names=("u", "g", "r", "i", "z", "y")):
    "Add approximate rest-frame time scales and distance-corrected peak-brightness proxies; these are not complete physical absolute magnitudes."
    t = result_table.copy()
    if "hostgal_photoz" in t.columns:
        uno_mas_z = (1.0 + t["hostgal_photoz"].clip(lower=0.0)).replace(0.0, 1.0)
        for b in band_names:
            for base in ("gp_rise_time", "gp_fall_time", "gp_fwhm"):
                col = f"{base}_{b}"
                if col in t.columns:
                    t[f"{col}_rest"] = t[col] / uno_mas_z
    if "distmod" in t.columns:
        for b in band_names:
            col = f"gp_peak_flux_{b}"
            if col in t.columns:
                flux = pd.to_numeric(t[col], errors="coerce")
                distmod = pd.to_numeric(t["distmod"], errors="coerce")
                valida = flux.gt(0.0) & distmod.gt(0.0) & np.isfinite(flux) & np.isfinite(distmod)
                magnitud = pd.Series(0.0, index=t.index, dtype=float)
                magnitud.loc[valida] = (
                    FLUX_ZEROPOINT - 2.5 * np.log10(flux.loc[valida]) - distmod.loc[valida]
                )
                t[f"gp_peak_absmag_{b}"] = magnitud

    new_indices = [c for c in t.columns if c not in result_table.columns]
    t[new_indices] = t[new_indices].fillna(0.0)
    return t
