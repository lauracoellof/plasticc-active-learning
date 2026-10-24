"""Post-hoc physical diagnostics of known classes and simulated anomaly subtypes."""

import numpy as np

try:
    from src.interpretability.shap_utils import OKABE_ITO as _OI
except Exception:
    _OI = {
        "azul": "#0072B2",
        "naranja": "#E69F00",
        "verde": "#009E73",
        "rojo": "#D55E00",
        "gris": "#999999",
        "morado": "#CC79A7",
    }


SUBMODELOS_99 = {
    991: {"nombre": "uLens-Binary", "tipo": "Galactic (binary microlensing)", "galactico": True},
    992: {"nombre": "ILOT", "tipo": "Faint, long extragalactic transient", "galactico": False},
    993: {"nombre": "CART", "tipo": "Faint, fast extragalactic transient", "galactico": False},
    994: {"nombre": "PISN", "tipo": "Ultraluminous extragalactic transient", "galactico": False},
}
COLOR99 = {991: _OI["morado"], 992: _OI["azul"], 993: _OI["verde"], 994: _OI["rojo"]}
NOMBRE99 = {k: v["nombre"] for k, v in SUBMODELOS_99.items()}


PASSBANDS = {
    0: ("#6a3d9a", "u"),
    1: ("#1f78b4", "g"),
    2: ("#33a02c", "r"),
    3: ("#e31a1c", "i"),
    4: ("#ff7f00", "z"),
    5: ("#b15928", "y"),
}

DESC_ETIQUETA = {
    "novedad": "Novedad\n(1-conf)",
    "iso": "Isolation\nForest",
    "gp": "Residuos\nGP",
    "mag": "Cambios\nmagnitud",
}


def descriptor_scores(clf, X, gp_df, st_df):
    """Compute novelty, isolation, GP-residual and amplitude diagnostics on aligned inputs."""
    import pandas as pd
    from src.models import anomalies as an

    scores = {
        "novedad": an.confidence_novelty_score(clf.predict_proba(X)),
        "iso": an.score_isolation_forest(X),
        "gp": an.gp_residual_score(gp_df),
        "mag": an.amplitude_score(st_df),
    }
    out = pd.DataFrame(scores)
    for k in scores:
        out[k + "_rank"] = an.rank01(scores[k])
    return out


def property_table(df, col="true_target"):
    """Summarize interpretation descriptors by the supplied physical subtype label."""

    cols = [
        c
        for c in [
            "hostgal_photoz",
            "x_absmag_r",
            "x_dur_rest",
            "x_simetria",
            "x_decline_slope",
            "x_tail_frac",
        ]
        if c in df.columns
    ]
    t = df.groupby(col)[cols].median().round(3)
    t.insert(0, "nombre", [NOMBRE99.get(int(i), str(i)) for i in t.index])
    t.insert(1, "n", df.groupby(col).size())
    t.insert(
        2,
        "naturaleza",
        [
            "Galactico" if SUBMODELOS_99.get(int(i), {}).get("galactico") else "Extragalactico"
            for i in t.index
        ],
    )
    return t


def phase_diagram(train_df, anom_df, save=None, submodelos=(992, 993, 994)):
    """Compare selected anomaly subtypes with training classes in a descriptive property plane."""
    import matplotlib.pyplot as plt

    def _val(d):
        return d[
            (d.x_absmag_r < -5) & (d.x_absmag_r > -25) & (d.x_dur_rest > 3) & (d.x_dur_rest < 600)
        ]

    tv, av = _val(train_df), _val(anom_df)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(
        tv.x_dur_rest,
        tv.x_absmag_r,
        s=8,
        color="#cfcfcf",
        alpha=0.5,
        edgecolor="none",
        label="Known classes (training set)",
        zorder=1,
    )
    for sm in submodelos:
        s = av[av.true_target == sm]
        n_tot = int((anom_df.true_target == sm).sum())
        ax.scatter(
            s.x_dur_rest,
            s.x_absmag_r,
            s=55,
            color=COLOR99[sm],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label=f"{sm} - {NOMBRE99[sm]} (n={n_tot})",
        )
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Rest-frame duration (days)")
    ax.set_ylabel("Peak absolute magnitude $M_r$ (brighter $\\uparrow$)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, color="#e6e6e6", lw=0.7)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return ax


def subtype_redshifts(anom_df, save=None, col_z="hostgal_photoz"):
    """Plot the supplied photometric-redshift distribution by simulated anomaly subtype."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    codigos = sorted(anom_df.true_target.unique())
    for i, sm in enumerate(codigos):
        z = anom_df.loc[anom_df.true_target == sm, col_z].to_numpy()
        x = np.full(len(z), i) + rng.uniform(-0.13, 0.13, len(z))
        ax.scatter(x, z, s=42, color=COLOR99[sm], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.plot([i - 0.22, i + 0.22], [np.median(z)] * 2, color="k", lw=2, zorder=5)
    ax.set_xticks(range(len(codigos)))
    ax.set_xticklabels([f"{sm}\n{NOMBRE99[sm]}" for sm in codigos])
    ax.set_ylabel("Host photometric redshift $z$")
    ax.axhline(0.02, color="#888", ls="--", lw=1)
    ax.text(
        0.02,
        0.035,
        "Galactic (z$\\approx$0)",
        fontsize=8,
        color="#555",
        transform=ax.get_yaxis_transform(),
    )
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.7)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return ax


def descriptor_heatmap(rank_df, tt, save=None):
    """Plot average ranked anomaly descriptors by simulated subtype."""
    import matplotlib.pyplot as plt

    tt = np.asarray(tt)
    sort_order = ["novedad", "iso", "gp", "mag"]
    codigos = [c for c in (991, 992, 993, 994) if (tt == c).any()]
    M = np.array(
        [[np.median(rank_df.loc[tt == c, d + "_rank"]) for c in codigos] for d in sort_order]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(codigos)))
    ax.set_xticklabels([NOMBRE99[c] for c in codigos])
    ax.set_yticks(range(len(sort_order)))
    ax.set_yticklabels([DESC_ETIQUETA[d] for d in sort_order])
    ax.grid(False)
    for a in range(len(sort_order)):
        for b in range(len(codigos)):
            ax.text(
                b,
                a,
                f"{M[a, b]:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if M[a, b] > 0.7 else "#222",
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Median anomaly rank\n(0 = normal, 1 = most anomalous)")
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return M


def light_curves(lc_df, objetos_por_sm, save=None):
    """Plot selected object light curves grouped by the supplied anomaly subtype."""
    import matplotlib.pyplot as plt

    codigos = list(objetos_por_sm)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, sm in zip(axes.ravel(), codigos):
        oid = int(objetos_por_sm[sm])
        g = lc_df[lc_df.object_id == oid]
        t0 = g.mjd.min()
        for pb in sorted(g.passband.unique()):
            gg = g[g.passband == pb]
            col, nm = PASSBANDS[int(pb)]
            ax.errorbar(
                gg.mjd - t0,
                gg.flux,
                yerr=gg.flux_err,
                fmt="o",
                ms=3,
                lw=0.6,
                color=col,
                label=nm,
                alpha=0.8,
            )
        ax.set_title(f"{sm} - {NOMBRE99[sm]}  (id {oid})", fontsize=10)
        ax.set_xlabel("Days since first observation")
        ax.set_ylabel("Flux")
        ax.grid(True, color="#e6e6e6", lw=0.7)
    axes[0, 0].legend(ncol=6, fontsize=7, frameon=False, loc="best")
    fig.suptitle("Illustrative light curves for 99x subtypes", fontsize=12)
    fig.text(
        0.5, 0.01, "Each panel uses its own flux scale.", ha="center", fontsize=8, color="#555555"
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return axes
