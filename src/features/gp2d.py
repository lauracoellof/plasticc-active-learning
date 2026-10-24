"""Joint time-wavelength Gaussian-process interpolation and photometric descriptors.

The wavelength axis uses observer-frame effective band wavelengths. The fixed
photometric schema is shared by the classifier, caches and saved experiments."""

import warnings

import numpy as np
import pandas as pd

import george
from george import kernels
from scipy.optimize import minimize

PASSBAND_MAP = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}

WAVELENGTHS = {0: 3670.7, 1: 4826.9, 2: 6223.2, 3: 7546.0, 4: 8690.9, 5: 9710.3}
WAVE_LETRA = {v: PASSBAND_MAP[k] for k, v in WAVELENGTHS.items()}
LETRA_WAVE = {PASSBAND_MAP[k]: v for k, v in WAVELENGTHS.items()}
BANDS = ["u", "g", "r", "i", "z", "y"]
PARES_COLOR = [
    ("u", "g", "ug"),
    ("g", "r", "gr"),
    ("r", "i", "ri"),
    ("i", "z", "iz"),
    ("z", "y", "zy"),
]

WAVE_LS = 6000.0
LS_TIME0 = 20.0

LS_TIME_BOUNDS = (5.0, 70.0)
LOG_CONST_BOUNDS = (float(np.log(1e-4)), float(np.log(1e2)))
MIN_OBS = 8
MIN_BANDAS = 3
N_COARSE = 80
VENTANA = 80.0
PASO_FINO = 1.0
DCOLOR_DIAS = 15.0

FASES_COLOR = (-10.0, 10.0, 20.0, 30.0)
P_SHOULDER = 25.0
SECMAX_LO, SECMAX_HI = 8.0, 45.0
P_TAIL = 50.0
RANDOM_STATE = 42
_FILL = 0.0


GP2D_FEATURES = (
    [
        "gp2d_ls_time",
        "gp2d_amp",
        "gp2d_loglik",
        "gp2d_chi2",
        "gp2d_resid_std",
        "gp2d_t_peak",
        "gp2d_peak_flux",
        "gp2d_peak_band",
    ]
    + [f"gp2d_peakflux_{b}" for b in BANDS]
    + [f"gp2d_color_{n}" for _, _, n in PARES_COLOR]
    + ["gp2d_dcolor_gr", "gp2d_dcolor_ri"]
    + [f"gp2d_dtpeak_{b}" for b in ["u", "g", "i", "z", "y"]]
    + ["gp2d_rise_time", "gp2d_fall_time", "gp2d_fwhm", "gp2d_fluence_total", "gp2d_n_obs"]
    + [f"gp2d_color_gr_{t}" for t in ("m10", "p10", "p20", "p30")]
    + [f"gp2d_color_ri_{t}" for t in ("m10", "p10", "p20", "p30")]
    + [
        "gp2d_sed_slope",
        "gp2d_sed_curv",
        "gp2d_dsed_slope",
        "gp2d_secmax_i",
        "gp2d_secmax_phase_i",
        "gp2d_shoulder_i",
        "gp2d_shoulder_z",
    ]
    + [
        "gp2d_secbump_i",
        "gp2d_secbump_z",
        "gp2d_dm15_r",
        "gp2d_dm15_i",
        "gp2d_dm30_i",
        "gp2d_tail_r",
        "gp2d_tail_i",
    ]
)


def _color_mag(f1, f2, floor):
    """Return a logarithmic colour from two positive fluxes using the configured floor."""
    a = f1 if f1 > floor else floor
    b = f2 if f2 > floor else floor
    return float(-2.5 * np.log10(a / b))


def _delta_mag(f_peak, f_late, floor):
    """Return a peak-to-later flux magnitude difference using the configured floor."""
    return _color_mag(f_late, f_peak, floor)


def fit_gp2d(light_curve):
    "Fit a joint time-wavelength Matern-3/2 Gaussian process.\n\nNormalize flux and flux error by the maximum absolute flux. Fix the wavelength\nlength scale and optimize the temporal kernel parameters within their bounds.\nThe posterior is a photometric interpolator, not a physical transient model."
    c = light_curve.dropna(subset=["mjd", "flux", "flux_err"])
    c = c[c["passband"].isin(PASSBAND_MAP)]
    if len(c) < MIN_OBS or c["passband"].nunique() < MIN_BANDAS:
        return None

    t = c["mjd"].to_numpy(float)
    t0 = t.min()
    tt = t - t0
    wave = c["passband"].map(WAVELENGTHS).to_numpy(float)
    flux = c["flux"].to_numpy(float)
    ferr = c["flux_err"].to_numpy(float)

    scale = float(np.max(np.abs(flux)))
    if scale <= 0:
        return None
    fn = flux / scale
    fen = np.clip(ferr / scale, 1e-3, None)

    X = np.column_stack([tt, wave])
    kernel = np.var(fn) * kernels.Matern32Kernel([LS_TIME0**2, WAVE_LS**2], ndim=2)
    gp = george.GP(kernel)
    try:
        gp.compute(X, fen)
    except (np.linalg.LinAlgError, ValueError):
        return None

    for item_name in gp.get_parameter_names():
        if item_name.endswith("1_1"):
            gp.freeze_parameter(item_name)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(fn, quiet=True)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(fn, quiet=True)

    p0 = gp.get_parameter_vector()
    bounds = []
    for nm in gp.get_parameter_names():
        if nm.endswith("0_0"):
            bounds.append((2 * np.log(LS_TIME_BOUNDS[0]), 2 * np.log(LS_TIME_BOUNDS[1])))
        elif "log_constant" in nm:
            bounds.append(LOG_CONST_BOUNDS)
        else:
            bounds.append((None, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = minimize(neg_ln_like, p0, jac=grad_neg_ln_like, method="L-BFGS-B", bounds=bounds)
            gp.set_parameter_vector(res.x)
        except (np.linalg.LinAlgError, ValueError):
            gp.set_parameter_vector(p0)

    return {
        "gp": gp,
        "t0": t0,
        "scale": scale,
        "X": X,
        "fn": fn,
        "flux": flux,
        "ferr": ferr,
        "tt": tt,
        "n": len(c),
    }


def _predict(estimator_model, t_abs, letra):
    """Evaluate the joint GP mean at absolute times in a selected photometric band."""
    t_abs = np.atleast_1d(np.asarray(t_abs, float))
    Xp = np.column_stack([t_abs - estimator_model["t0"], np.full(t_abs.shape, LETRA_WAVE[letra])])
    mu = estimator_model["gp"].predict(
        estimator_model["fn"], Xp, return_cov=False, return_var=False
    )
    return mu * estimator_model["scale"]


def _lengthscale_and_amplitude(estimator_model):
    """Recover the fitted temporal length scale and kernel amplitude."""
    d = estimator_model["gp"].get_parameter_dict(include_frozen=True)
    log_M00 = next(v for k, v in d.items() if k.endswith("0_0"))
    log_const = next(
        v for k, v in d.items() if "log_constant" in k or k.endswith("k1:log_constant")
    )
    ls_time = float(np.exp(0.5 * log_M00))
    amp = float(np.sqrt(np.exp(log_const)) * estimator_model["scale"])
    return ls_time, amp


def _temporal_shape(t_grid, mu):
    """Summarize posterior-mean rise, decline and width on the evaluation grid."""
    if mu.size == 0 or not np.isfinite(mu).any():
        return _FILL, _FILL, _FILL
    i_pk = int(np.argmax(mu))
    peak_value = mu[i_pk]
    if peak_value <= 0:
        return _FILL, _FILL, _FILL
    media = peak_value / 2.0
    izq = np.where(mu[:i_pk] <= media)[0]
    t_rise = t_grid[i_pk] - t_grid[izq[-1]] if izq.size else _FILL
    der = np.where(mu[i_pk:] <= media)[0]
    t_fall = t_grid[i_pk + der[0]] - t_grid[i_pk] if der.size else _FILL
    fwhm = (t_rise + t_fall) if (izq.size and der.size) else _FILL
    return float(t_rise), float(t_fall), float(fwhm)


def _asinh_mag(f, soft):
    """Map signed flux to a smooth asinh brightness coordinate."""
    return -2.5 / np.log(10) * (np.arcsinh(np.asarray(f, float) / (2 * soft)) + np.log(soft))


def _sed_coef(estimator_model, t_abs, soft):
    """Fit low-order spectral-shape coefficients to the GP band predictions at one epoch."""
    fl = np.array([_predict(estimator_model, t_abs, b)[0] for b in BANDS])
    mag = _asinh_mag(fl, soft)
    logw = np.log10(np.array([LETRA_WAVE[b] for b in BANDS]))
    c = np.polyfit(logw - logw.mean(), mag, 2)
    return float(c[1]), float(2.0 * c[0])


def _color_at_phase(estimator_model, t_peak_abs, phase_value, b1, b2, floor, span):
    """Evaluate a band-pair colour at a supported phase relative to the reference peak."""
    t = t_peak_abs + phase_value
    if not (estimator_model["t0"] <= t <= estimator_model["t0"] + span):
        return _FILL
    return _color_mag(
        float(_predict(estimator_model, t, b1)[0]),
        float(_predict(estimator_model, t, b2)[0]),
        floor,
    )


def _secondary_maximum(t_grid, mu):
    """Estimate the relative height and timing of a supported post-peak local maximum."""
    if mu.size < 5 or not np.isfinite(mu).any():
        return _FILL, _FILL
    i_pri = int(np.argmax(mu))
    f_pri = mu[i_pri]
    if f_pri <= 0:
        return _FILL, _FILL
    m = (t_grid >= t_grid[i_pri] + SECMAX_LO) & (t_grid <= t_grid[i_pri] + SECMAX_HI)
    idx = np.where(m)[0]
    if idx.size < 3:
        return _FILL, _FILL
    best_f, best_t = 0.0, 0.0
    for k in idx[1:-1]:
        if mu[k] > mu[k - 1] and mu[k] >= mu[k + 1] and mu[k] > best_f:
            best_f, best_t = mu[k], t_grid[k]
    if best_f <= 0:
        return _FILL, _FILL
    return float(best_f / f_pri), float(best_t - t_grid[i_pri])


def _secbump(t_grid, mu):
    """Measure a late-time bump relative to the primary GP posterior-mean peak."""
    if mu.size < 5 or not np.isfinite(mu).any():
        return _FILL
    i_pri = int(np.argmax(mu))
    f_pri = mu[i_pri]
    if f_pri <= 0:
        return _FILL
    m = (t_grid >= t_grid[i_pri] + 10.0) & (t_grid <= t_grid[i_pri] + 55.0)
    idx = np.where(m)[0]
    if idx.size < 3:
        return _FILL
    seg = mu[idx]
    j_tr = int(np.argmin(seg))
    f_sec = float(np.max(seg[j_tr:]))
    return float(max(0.0, f_sec - seg[j_tr]) / f_pri)


def object_gp2d_features(light_curve):
    """Return the fixed GP2D descriptor dictionary for one multiband light curve."""
    out = {n: _FILL for n in GP2D_FEATURES}
    estimator_model = fit_gp2d(light_curve)
    if estimator_model is None:
        return out

    ls_time, amp = _lengthscale_and_amplitude(estimator_model)
    mu_obs = (
        estimator_model["gp"].predict(
            estimator_model["fn"], estimator_model["X"], return_cov=False, return_var=False
        )
        * estimator_model["scale"]
    )
    resid = (estimator_model["flux"] - mu_obs) / estimator_model["ferr"]
    out["gp2d_ls_time"] = ls_time
    out["gp2d_amp"] = amp
    out["gp2d_loglik"] = float(
        estimator_model["gp"].log_likelihood(estimator_model["fn"], quiet=True)
        / estimator_model["n"]
    )
    out["gp2d_chi2"] = float(np.mean(resid**2))
    out["gp2d_resid_std"] = float(np.std(resid))

    span = float(estimator_model["tt"].max())
    t_coarse = np.linspace(0.0, span, N_COARSE)
    coarse = {b: _predict(estimator_model, t_coarse + estimator_model["t0"], b) for b in BANDS}
    pb0 = max(BANDS, key=lambda b: float(np.max(coarse[b])))
    t_peak0 = float(t_coarse[int(np.argmax(coarse[pb0]))])

    lo, hi = max(0.0, t_peak0 - VENTANA), min(span, t_peak0 + VENTANA)
    t_grid = np.arange(lo, hi + 1e-6, PASO_FINO)
    if t_grid.size < 3:
        t_grid = np.linspace(lo, max(hi, lo + 1.0), 3)
    band_curve = {b: _predict(estimator_model, t_grid + estimator_model["t0"], b) for b in BANDS}
    peak_flux_b = {b: float(np.max(band_curve[b])) for b in BANDS}
    peak_t_b = {b: float(t_grid[int(np.argmax(band_curve[b]))]) for b in BANDS}

    pb = max(peak_flux_b, key=peak_flux_b.get)
    t_peak = peak_t_b[pb]
    out["gp2d_t_peak"] = t_peak
    out["gp2d_peak_flux"] = peak_flux_b[pb]
    out["gp2d_peak_band"] = float(BANDS.index(pb))

    flux_peak = {
        b: float(_predict(estimator_model, t_peak + estimator_model["t0"], b)[0]) for b in BANDS
    }
    for b in BANDS:
        out[f"gp2d_peakflux_{b}"] = flux_peak[b]

    floor = max(0.01 * abs(peak_flux_b[pb]), 1e-3)
    for b1, b2, n in PARES_COLOR:
        out[f"gp2d_color_{n}"] = _color_mag(flux_peak[b1], flux_peak[b2], floor)

    if t_peak + DCOLOR_DIAS <= span:
        flux_post = {
            b: float(_predict(estimator_model, t_peak + DCOLOR_DIAS + estimator_model["t0"], b)[0])
            for b in ["g", "r", "i"]
        }
        out["gp2d_dcolor_gr"] = (
            _color_mag(flux_post["g"], flux_post["r"], floor) - out["gp2d_color_gr"]
        )
        out["gp2d_dcolor_ri"] = (
            _color_mag(flux_post["r"], flux_post["i"], floor) - out["gp2d_color_ri"]
        )

    for b in ["u", "g", "i", "z", "y"]:
        out[f"gp2d_dtpeak_{b}"] = peak_t_b[b] - peak_t_b["r"]

    out["gp2d_rise_time"], out["gp2d_fall_time"], out["gp2d_fwhm"] = _temporal_shape(
        t_grid, band_curve["r"]
    )
    out["gp2d_fluence_total"] = float(
        sum(np.trapezoid(np.clip(band_curve[b], 0, None), t_grid) for b in BANDS)
    )
    out["gp2d_n_obs"] = float(estimator_model["n"])

    t_peak_abs = t_peak + estimator_model["t0"]

    TAGS = {-10.0: "m10", 10.0: "p10", 20.0: "p20", 30.0: "p30"}
    for phase_value in FASES_COLOR:
        tag = TAGS[phase_value]
        out[f"gp2d_color_gr_{tag}"] = _color_at_phase(
            estimator_model, t_peak_abs, phase_value, "g", "r", floor, span
        )
        out[f"gp2d_color_ri_{tag}"] = _color_at_phase(
            estimator_model, t_peak_abs, phase_value, "r", "i", floor, span
        )

    s0, k0 = _sed_coef(estimator_model, t_peak_abs, floor)
    out["gp2d_sed_slope"], out["gp2d_sed_curv"] = s0, k0
    if estimator_model["t0"] <= t_peak_abs + 20.0 <= estimator_model["t0"] + span:
        s20, _k = _sed_coef(estimator_model, t_peak_abs + 20.0, floor)
        out["gp2d_dsed_slope"] = s20 - s0

    out["gp2d_secmax_i"], out["gp2d_secmax_phase_i"] = _secondary_maximum(t_grid, band_curve["i"])
    for b in ("i", "z"):
        t_sh = t_peak_abs + P_SHOULDER
        if peak_flux_b[b] > 0 and estimator_model["t0"] <= t_sh <= estimator_model["t0"] + span:
            out[f"gp2d_shoulder_{b}"] = float(
                _predict(estimator_model, t_sh, b)[0] / peak_flux_b[b]
            )

    out["gp2d_secbump_i"] = _secbump(t_grid, band_curve["i"])
    out["gp2d_secbump_z"] = _secbump(t_grid, band_curve["z"])

    for b, dias, item_name in (
        ("r", 15.0, "gp2d_dm15_r"),
        ("i", 15.0, "gp2d_dm15_i"),
        ("i", 30.0, "gp2d_dm30_i"),
    ):
        tp = peak_t_b[b] + estimator_model["t0"]
        if estimator_model["t0"] <= tp + dias <= estimator_model["t0"] + span:
            f_late = float(_predict(estimator_model, tp + dias, b)[0])
            out[item_name] = _delta_mag(peak_flux_b[b], f_late, floor)

    for b in ("r", "i"):
        tp = peak_t_b[b] + estimator_model["t0"]
        if (
            peak_flux_b[b] > 0
            and estimator_model["t0"] <= tp + P_TAIL <= estimator_model["t0"] + span
        ):
            out[f"gp2d_tail_{b}"] = float(
                np.clip(_predict(estimator_model, tp + P_TAIL, b)[0], 0, None) / peak_flux_b[b]
            )

    return out


def build_gp2d_table(df, progress_interval=500):
    """Fit joint time-wavelength descriptors and return one row per object_id."""
    records_list = []
    object_ids = df["object_id"].unique()
    for k, oid in enumerate(object_ids):
        feats = object_gp2d_features(df[df["object_id"] == oid])
        feats["object_id"] = oid
        records_list.append(feats)
        if progress_interval and (k + 1) % progress_interval == 0:
            print(f"  {k + 1}/{len(object_ids)} objects")
    cols = ["object_id"] + GP2D_FEATURES
    return pd.DataFrame(records_list)[cols]
