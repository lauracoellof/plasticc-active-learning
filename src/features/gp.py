"""Per-band Gaussian-process interpolation and light-curve descriptors.

Times are MJD days and fluxes follow PLAsTiCC units. Unsupported fits use the
historical zero-filled feature schema; GP posteriors are interpolation models."""

import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

PASSBAND_MAP = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}


MIN_OBS_BANDA = 5
RANDOM_STATE = 42


LS_MIN = 10.0


GP_FEATURES = [
    "gp_lenscale",
    "gp_amp",
    "gp_noise",
    "gp_loglik",
    "gp_chi2",
    "gp_resid_std",
    "gp_resid_max",
    "gp_outfrac",
    "gp_fluence",
    "gp_fluence_norm",
    "gp_t_peak",
    "gp_rise_time",
    "gp_fall_time",
    "gp_rise_rate",
    "gp_fall_rate",
    "gp_fwhm",
    "gp_peak_flux",
    "gp_rise_fall_ratio",
    "gp_dm15",
    "gp_dm30",
]


N_GRID_FLUENCE = 200


_FILL = 0.0


def _build_kernel():
    """Construct the per-band constant-times-Matern kernel with an additive white-noise term."""
    amplitud = ConstantKernel(1.0, (1e-3, 1e3))
    forma = Matern(length_scale=20.0, length_scale_bounds=(LS_MIN, 1e3), nu=1.5)
    ruido = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    return amplitud * forma + ruido


def fit_object_gp(light_curve, band_index):
    "Fit a one-dimensional GP to one object's observations in one band.\n\nUse a constant times Matern-3/2 kernel plus white noise. Include the scaled\nflux errors as observational variances. Return no fit when support is insufficient."
    sub = light_curve.dropna(subset=["mjd", "flux"]).sort_values("mjd")
    n = len(sub)
    if n < MIN_OBS_BANDA:
        return None

    t = sub["mjd"].to_numpy(dtype=float)
    f = sub["flux"].to_numpy(dtype=float)
    ferr = sub["flux_err"].to_numpy(dtype=float) if "flux_err" in sub else np.zeros(n)

    t0 = t.min()
    X = (t - t0).reshape(-1, 1)

    flux_mean = f.mean()
    flux_std = f.std(ddof=1)
    if not np.isfinite(flux_std) or flux_std < 1e-8:
        flux_std = 1.0
    y = (f - flux_mean) / flux_std

    alpha = (ferr / flux_std) ** 2
    alpha = np.where(np.isfinite(alpha), alpha, 0.0) + 1e-6

    gp = GaussianProcessRegressor(
        kernel=_build_kernel(),
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=1,
        random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            gp.fit(X, y)
        except Exception:
            return None

    return {
        "gp": gp,
        "t0": t0,
        "flux_mean": flux_mean,
        "flux_std": flux_std,
        "n": n,
        "X": X,
        "y": y,
    }


def _shape_descriptors(t_grid, mu_grid):
    """Summarize rise, decline and width on a GP posterior-mean time grid."""
    i = int(np.argmax(mu_grid))
    t_peak = float(t_grid[i] - t_grid[0])
    peak = float(mu_grid[i])
    base = float(mu_grid.min())
    amp = peak - base

    nulo = {
        "gp_t_peak": t_peak,
        "gp_rise_time": 0.0,
        "gp_fall_time": 0.0,
        "gp_rise_rate": 0.0,
        "gp_fall_rate": 0.0,
        "gp_fwhm": 0.0,
    }
    if amp <= 1e-9:
        return nulo

    half = base + amp / 2.0

    izq = np.where(mu_grid[: i + 1] <= half)[0]
    t_rise_start = float(t_grid[izq[-1]]) if izq.size else float(t_grid[0])
    rise_time = float(t_grid[i] - t_rise_start)

    der = np.where(mu_grid[i:] <= half)[0]
    t_fall_end = float(t_grid[i + der[0]]) if der.size else float(t_grid[-1])
    fall_time = float(t_fall_end - t_grid[i])

    rise_rate = amp / rise_time if rise_time > 1e-6 else 0.0
    fall_rate = -amp / fall_time if fall_time > 1e-6 else 0.0
    fwhm = rise_time + fall_time
    return {
        "gp_t_peak": t_peak,
        "gp_rise_time": rise_time,
        "gp_fall_time": fall_time,
        "gp_rise_rate": rise_rate,
        "gp_fall_rate": fall_rate,
        "gp_fwhm": fwhm,
    }


def _band_features(fit_result):
    """Convert a supported per-band GP fit into the fixed descriptor dictionary."""
    gp = fit_result["gp"]
    X, y = fit_result["X"], fit_result["y"]
    flux_std = fit_result["flux_std"]

    k = gp.kernel_
    const_k, matern_k, white_k = k.k1.k1, k.k1.k2, k.k2
    lenscale = float(np.ravel(matern_k.length_scale)[0])
    amp = float(np.sqrt(const_k.constant_value)) * flux_std
    noise = float(white_k.noise_level)
    loglik = float(gp.log_marginal_likelihood_value_) / fit_result["n"]

    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.where(sigma > 1e-6, sigma, 1e-6)
    resid = (y - mu) / sigma
    chi2 = float(np.mean(resid**2))
    resid_std = float(np.std(resid))
    resid_max = float(np.max(np.abs(resid)))
    outfrac = float(np.mean(np.abs(resid) > 3.0))

    t_grid = np.linspace(float(X.min()), float(X.max()), N_GRID_FLUENCE)
    mu_grid = gp.predict(t_grid.reshape(-1, 1)) * flux_std + fit_result["flux_mean"]
    duracion = float(X.max() - X.min())
    fluence = float(np.trapezoid(mu_grid, t_grid))
    fluence_norm = fluence / duracion if duracion > 1e-6 else 0.0

    out = {
        "gp_lenscale": lenscale,
        "gp_amp": amp,
        "gp_noise": noise,
        "gp_loglik": loglik,
        "gp_chi2": chi2,
        "gp_resid_std": resid_std,
        "gp_resid_max": resid_max,
        "gp_outfrac": outfrac,
        "gp_fluence": fluence,
        "gp_fluence_norm": fluence_norm,
    }
    out.update(_shape_descriptors(t_grid, mu_grid))
    out["gp_peak_flux"] = float(np.max(mu_grid))
    _ft = out["gp_fall_time"]
    out["gp_rise_fall_ratio"] = out["gp_rise_time"] / _ft if _ft > 1e-6 else 0.0

    _pk = out["gp_peak_flux"]
    _tpk = out["gp_t_peak"]
    _floor_b = max(0.05 * abs(_pk), 1e-3)

    def _dmag(dt):
        if _tpk + dt > float(X.max()):
            return _FILL
        x = np.array([[_tpk + dt]])
        fp = float(gp.predict(x)[0] * flux_std + fit_result["flux_mean"])
        return _delta_mag(_pk, fp, _floor_b)

    out["gp_dm15"] = _dmag(15.0)
    out["gp_dm30"] = _dmag(30.0)
    out["_resid"] = resid
    return out


_PARES_COLOR = [(1, 2, "gr"), (2, 3, "ri"), (3, 4, "iz")]
DCOLOR_DIAS = 20.0


GP_FEATURES_COLOR = (
    [f"gp_color_{n}_peak" for _, _, n in _PARES_COLOR]
    + [f"gp_dcolor_{n}" for _, _, n in _PARES_COLOR]
    + [f"gp_peaklag_{n}" for _, _, n in _PARES_COLOR]
)


def _flux_at(fit_result, t_abs):
    """Interpolate the fitted mean at an absolute MJD, returning NaN outside support."""
    x = np.array([[t_abs - fit_result["t0"]]])
    mu = fit_result["gp"].predict(x)[0] * fit_result["flux_std"] + fit_result["flux_mean"]
    return float(mu)


def _within_support(fit_result, t_abs):
    """Test whether an absolute MJD falls within the fitted observation interval."""
    t_rel = float(t_abs - fit_result["t0"])
    return float(fit_result["X"].min()) <= t_rel <= float(fit_result["X"].max())


def _color_mag(f1, f2, floor):
    """Return a logarithmic colour from two positive fluxes using the configured floor."""
    a = f1 if f1 > floor else floor
    b = f2 if f2 > floor else floor
    return float(-2.5 * np.log10(a / b))


def _delta_mag(f_peak, f_late, floor):
    """Return a peak-to-later flux magnitude difference using the configured floor."""
    return _color_mag(f_late, f_peak, floor)


def _color_features(ajustes, peak_flux_b, peak_mjd_b):
    """Build cross-band colours and colour evolution from the fitted per-band curves."""
    out = {n: _FILL for n in GP_FEATURES_COLOR}
    if not ajustes:
        return out

    pb_peak = max(peak_flux_b, key=peak_flux_b.get)
    t_peak = peak_mjd_b[pb_peak]
    floor = max(0.01 * abs(peak_flux_b[pb_peak]), 1e-3)
    for b1, b2, n in _PARES_COLOR:
        if b1 in ajustes and b2 in ajustes:
            out[f"gp_peaklag_{n}"] = float(peak_mjd_b[b1] - peak_mjd_b[b2])

            if _within_support(ajustes[b1], t_peak) and _within_support(ajustes[b2], t_peak):
                f1 = _flux_at(ajustes[b1], t_peak)
                f2 = _flux_at(ajustes[b2], t_peak)
                out[f"gp_color_{n}_peak"] = _color_mag(f1, f2, floor)
                t_post = t_peak + DCOLOR_DIAS
                if _within_support(ajustes[b1], t_post) and _within_support(ajustes[b2], t_post):
                    g1 = _flux_at(ajustes[b1], t_post)
                    g2 = _flux_at(ajustes[b2], t_post)
                    out[f"gp_dcolor_{n}"] = _color_mag(g1, g2, floor) - out[f"gp_color_{n}_peak"]
    return out


def object_gp_features(light_curve):
    """Build the fixed GP1D descriptor vector for one object across all six bands."""
    out = {}
    chi2_list, residmax_list, outfrac_list = [], [], []
    fluence_list, fwhm_list, rise_list = [], [], []
    n_ok = 0
    ajustes = {}
    peak_flux_b = {}
    peak_mjd_b = {}

    for pb, letra in PASSBAND_MAP.items():
        band_index = light_curve[light_curve["passband"] == pb]
        fit_result = fit_object_gp(band_index, pb)
        if fit_result is None:
            for feat in GP_FEATURES:
                out[f"{feat}_{letra}"] = _FILL
            continue
        f = _band_features(fit_result)
        for feat in GP_FEATURES:
            out[f"{feat}_{letra}"] = f[feat]
        chi2_list.append(f["gp_chi2"])
        residmax_list.append(f["gp_resid_max"])
        outfrac_list.append(f["gp_outfrac"])
        fluence_list.append(f["gp_fluence"])
        fwhm_list.append(f["gp_fwhm"])
        rise_list.append(f["gp_rise_time"])
        ajustes[pb] = fit_result
        peak_flux_b[pb] = f["gp_peak_flux"]
        peak_mjd_b[pb] = fit_result["t0"] + f["gp_t_peak"]
        n_ok += 1

    out["gp_chi2_max"] = float(np.max(chi2_list)) if chi2_list else _FILL
    out["gp_resid_max_global"] = float(np.max(residmax_list)) if residmax_list else _FILL
    out["gp_outfrac_mean"] = float(np.mean(outfrac_list)) if outfrac_list else _FILL
    out["gp_fluence_total"] = float(np.sum(fluence_list)) if fluence_list else _FILL
    out["gp_fwhm_mean"] = float(np.mean(fwhm_list)) if fwhm_list else _FILL
    out["gp_rise_time_mean"] = float(np.mean(rise_list)) if rise_list else _FILL
    out["gp_n_bandas_ok"] = float(n_ok)

    out.update(_color_features(ajustes, peak_flux_b, peak_mjd_b))
    return out


def object_gp_residuals(light_curve):
    """Summarize observed-minus-predicted residuals for each supported per-band fit."""
    residuos = []
    for pb in PASSBAND_MAP:
        band_index = light_curve[light_curve["passband"] == pb]
        fit_result = fit_object_gp(band_index, pb)
        if fit_result is None:
            continue
        residuos.append(_band_features(fit_result)["_resid"])
    if not residuos:
        return np.array([], dtype=float)
    return np.concatenate(residuos)


def build_gp_table(df, progress_interval=500):
    """Fit object-by-object GP1D descriptors and return an object_id-keyed feature table."""
    records_list = []
    object_ids = df["object_id"].unique()
    total = len(object_ids)
    for i, (oid, light_curve) in enumerate(df.groupby("object_id", sort=False), 1):
        fila = {"object_id": oid}
        fila.update(object_gp_features(light_curve))
        records_list.append(fila)
        if progress_interval and (i % progress_interval == 0 or i == total):
            print(f"  GP: {i}/{total} objects processed")
    result_table = pd.DataFrame(records_list)
    feat_cols = [c for c in result_table.columns if c != "object_id"]
    result_table[feat_cols] = result_table[feat_cols].fillna(_FILL)
    return result_table
