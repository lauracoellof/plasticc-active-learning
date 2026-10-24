"""Post-hoc period, shape and brightness proxies for physical interpretation.

These descriptors are not additional inputs to the canonical 304-feature classifier.
Zero is the historical missing/unsupported-value convention, not a physical measurement."""

import numpy as np
import pandas as pd

PASSBAND_MAP = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
_FILL = 0.0
_EPS = 1e-6

EXTRA_FEATURES = [
    "x_ls_period",
    "x_ls_power",
    "x_ls_nciclos",
    "x_simetria",
    "x_acromaticidad",
    "x_decline_slope",
    "x_tail_frac",
    "x_absmag_g",
    "x_absmag_r",
    "x_absmag_i",
    "x_color_gr_peak",
    "x_color_ri_peak",
    "x_dur_rest",
]


_P_MIN = 0.2
_ZP = 27.5


def _lomb_scargle(t, y, dy, baseline):
    """Return the strongest Lomb-Scargle period and power, or zero for unsupported fits."""
    if len(t) < 10 or baseline <= 0:
        return _FILL, _FILL
    try:
        from astropy.timeseries import LombScargle
    except Exception:
        return _FILL, _FILL
    try:
        fmin = 1.0 / baseline
        fmax = 1.0 / _P_MIN
        dy_safe = np.where(dy > 0, dy, np.nanmedian(dy[dy > 0]) if np.any(dy > 0) else 1.0)
        freq, power = LombScargle(t, y, dy_safe).autopower(
            minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=5
        )
        if len(power) == 0:
            return _FILL, _FILL
        i = int(np.argmax(power))
        return float(1.0 / freq[i]), float(power[i])
    except Exception:
        return _FILL, _FILL


def _symmetry(t, y):
    """Compare the flux-weighted time centroid with the peak; return a bounded shape proxy."""
    if len(t) < 5:
        return _FILL
    span = float(t.max() - t.min())
    if span <= 0:
        return _FILL
    f = y - y.min()
    s = f.sum()
    if s <= _EPS:
        return _FILL
    t_centroide = float((f * t).sum() / s)
    t_pico = float(t[np.argmax(y)])
    return float(np.clip(1.0 - 2.0 * abs(t_centroide - t_pico) / span, 0.0, 1.0))


def _achromaticity(grp, nbins=10):
    """Average pairwise band correlations after binning and normalizing their flux curves."""
    t_all = grp["mjd"].to_numpy(float)
    if len(t_all) < 8:
        return _FILL
    t0, t1 = t_all.min(), t_all.max()
    if t1 <= t0:
        return _FILL
    edges = np.linspace(t0, t1, nbins + 1)
    curvas = {}
    for b, sub in grp.groupby("passband"):
        if len(sub) < 4:
            continue
        idx = np.clip(np.digitize(sub["mjd"].to_numpy(float), edges) - 1, 0, nbins - 1)
        v = np.full(nbins, np.nan)
        f = sub["flux"].to_numpy(float)
        for k in range(nbins):
            m = idx == k
            if m.any():
                v[k] = f[m].mean()
        peak_value = np.nanmax(np.abs(v))
        if peak_value > _EPS:
            curvas[b] = v / peak_value
    band_names = list(curvas)
    if len(band_names) < 2:
        return _FILL
    corrs = []
    for i in range(len(band_names)):
        for j in range(i + 1, len(band_names)):
            a, c = curvas[band_names[i]], curvas[band_names[j]]
            com = ~np.isnan(a) & ~np.isnan(c)
            if com.sum() >= 4 and np.std(a[com]) > _EPS and np.std(c[com]) > _EPS:
                corrs.append(np.corrcoef(a[com], c[com])[0, 1])
    return float(np.mean(corrs)) if corrs else _FILL


def _decline(t, y):
    """Return the post-peak log-log slope and late-time flux fraction when supported."""
    if len(t) < 6:
        return _FILL, _FILL
    ip = int(np.argmax(y))
    peak_value = float(y[ip])
    if peak_value <= _EPS:
        return _FILL, _FILL
    tp = t[ip]
    post = t > tp
    dt = t[post] - tp
    fl = y[post]
    pos = fl > 0
    pendiente = _FILL
    if pos.sum() >= 4:
        x = np.log10(dt[pos] + 1.0)
        z = np.log10(fl[pos])
        if np.std(x) > _EPS:
            pendiente = float(np.polyfit(x, z, 1)[0])

    span = float(t.max() - t.min())
    cola = y[t >= t.min() + 0.75 * span]
    tail_frac = float(np.median(cola) / peak_value) if len(cola) else _FILL
    return pendiente, tail_frac


def _absmag_color_dur(grp, z, distmod):
    """Compute peak-brightness, colour and time-dilation proxies from observed light curves.

    Brightness proxies use the PLAsTiCC zero point and distance modulus; they omit
    extinction and K corrections and must not be read as full physical absolute magnitudes."""
    out = {
        "x_absmag_g": _FILL,
        "x_absmag_r": _FILL,
        "x_absmag_i": _FILL,
        "x_color_gr_peak": _FILL,
        "x_color_ri_peak": _FILL,
        "x_dur_rest": _FILL,
    }
    pk = {}
    for b in (1, 2, 3):
        s = grp.loc[grp["passband"] == b, "flux"]
        if len(s) >= 3 and s.max() > 0:
            pk[b] = float(s.max())

    if distmod > 0:
        if 1 in pk:
            out["x_absmag_g"] = _ZP - 2.5 * np.log10(pk[1]) - distmod
        if 2 in pk:
            out["x_absmag_r"] = _ZP - 2.5 * np.log10(pk[2]) - distmod
        if 3 in pk:
            out["x_absmag_i"] = _ZP - 2.5 * np.log10(pk[3]) - distmod
    if 1 in pk and 2 in pk:
        out["x_color_gr_peak"] = -2.5 * np.log10(pk[1] / pk[2])
    if 2 in pk and 3 in pk:
        out["x_color_ri_peak"] = -2.5 * np.log10(pk[2] / pk[3])
    span = float(grp["mjd"].max() - grp["mjd"].min())
    out["x_dur_rest"] = span / (1.0 + max(z, 0.0))
    return out


def _group_extra_features(grp, z=0.0, distmod=0.0):
    """Build post-hoc descriptors for one object, using its most-sampled band for shape."""

    conteo = grp.groupby("passband").size()
    b_main = int(conteo.idxmax())
    sub = grp[grp["passband"] == b_main].sort_values("mjd")
    t = sub["mjd"].to_numpy(float)
    y = sub["flux"].to_numpy(float)
    dy = sub["flux_err"].to_numpy(float)
    baseline = float(t.max() - t.min()) if len(t) > 1 else 0.0

    periodo, potencia = _lomb_scargle(t, y, dy, baseline)
    nciclos = float(baseline / periodo) if periodo > 0 else _FILL
    pendiente, tail = _decline(t, y)

    out = {
        "x_ls_period": periodo,
        "x_ls_power": potencia,
        "x_ls_nciclos": nciclos,
        "x_simetria": _symmetry(t, y),
        "x_acromaticidad": _achromaticity(grp),
        "x_decline_slope": pendiente,
        "x_tail_frac": tail,
    }
    out.update(_absmag_color_dur(grp, z, distmod))
    return pd.Series(out)


def build_extra_feature_table(df, metadata_table=None, progress_interval=500):
    """Return one row of interpretation descriptors per object; fill unsupported values with zero."""
    if metadata_table is not None:
        zmap = metadata_table.set_index("object_id")["hostgal_photoz"].to_dict()
        dmap = metadata_table.set_index("object_id")["distmod"].fillna(0.0).to_dict()
    else:
        zmap, dmap = {}, {}

    records_list, object_ids = [], []
    for k, (oid, grp) in enumerate(df.groupby("object_id"), 1):
        records_list.append(
            _group_extra_features(
                grp, float(zmap.get(oid, 0.0) or 0.0), float(dmap.get(oid, 0.0) or 0.0)
            )
        )
        object_ids.append(oid)
        if progress_interval and k % progress_interval == 0:
            print(f"  extra: {k:,} objects processed")
    result_table = pd.DataFrame(records_list)
    result_table.insert(0, "object_id", object_ids)
    return result_table.fillna(_FILL)
