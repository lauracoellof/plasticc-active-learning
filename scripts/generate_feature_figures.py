"Generate feature figures."

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.features import gp, gp2d


OUT_REPO = ROOT / "reports" / "figures"
OUT_REPO.mkdir(parents=True, exist_ok=True)


def save(fig, item_name):
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_REPO / f"{item_name}.{suffix}", bbox_inches="tight", facecolor="white")


def example_curves(df, metadata_table):
    display_names = {90: "SNIa", 16: "eclipsing binary", 92: "RR Lyrae"}
    band_names = list("ugrizy")
    colores = {0: "#6a3d9a", 1: "#1b9e77", 2: "#d62728", 3: "#8c2d04", 4: "#7f5539", 5: "#525252"}
    ejemplos = {}
    for class_code in display_names:
        object_ids = metadata_table.loc[metadata_table.target == class_code, "object_id"]
        sub = df[df.object_id.isin(object_ids)]
        ejemplos[class_code] = int(sub.groupby("object_id").size().idxmax())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (class_code, oid) in zip(axes, ejemplos.items()):
        light_curve = df[df.object_id == oid]
        for band_index in range(6):
            s = light_curve[light_curve.passband == band_index]
            ax.errorbar(
                s.mjd,
                s.flux,
                yerr=s.flux_err,
                fmt="o",
                ms=3,
                lw=0.6,
                color=colores[band_index],
                alpha=0.72,
                label=band_names[band_index],
            )
        ax.axhline(0, color="0.25", lw=0.6, ls="--")
        ax.set_title(f"{display_names[class_code]} · object {oid}")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Differential flux")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=7, title="Band")
    fig.tight_layout()
    save(fig, "01_curvas_ejemplo")
    plt.close(fig)


def gp1d_fit(df, oid=252646):
    light_curve = df[df.object_id == oid]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2), sharex=True)
    for pb, letra in gp.PASSBAND_MAP.items():
        ax = axes.ravel()[pb]
        band_index = light_curve[light_curve.passband == pb]
        fit_result = gp.fit_object_gp(band_index, pb)
        if fit_result is None:
            ax.set_title(f"Band {letra}: <{gp.MIN_OBS_BANDA} observations")
            ax.set_axis_off()
            continue
        estimator_model, t0 = fit_result["gp"], fit_result["t0"]
        t = band_index.mjd.values - t0
        xx = np.linspace(t.min(), t.max(), 200).reshape(-1, 1)
        mu, sd = estimator_model.predict(xx, return_std=True)
        mu = mu * fit_result["flux_std"] + fit_result["flux_mean"]
        sd = sd * fit_result["flux_std"]
        integral = np.trapezoid(mu, xx.ravel())
        ipk = int(np.argmax(mu))
        ax.fill_between(xx.ravel(), 0, mu, color="C2", alpha=0.18)
        ax.errorbar(
            t,
            band_index.flux,
            yerr=band_index.flux_err,
            fmt=".",
            ms=4,
            alpha=0.6,
            label="Observaciones",
        )
        ax.plot(xx.ravel(), mu, "-", color="C3", lw=1.5, label="GP mean")
        ax.fill_between(xx.ravel(), mu - 2 * sd, mu + 2 * sd, color="C3", alpha=0.15)
        ax.plot(xx.ravel()[ipk], mu[ipk], "o", color="C3", ms=6, label="Peak")
        ax.axhline(0, color="0.25", lw=0.5)
        ax.set_title(f"Band {letra} · integral={integral:,.0f}")
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"Object {oid}: independent per-band GP1D fits", y=0.995)
    fig.supxlabel("Days since first observation")
    fig.supylabel("Differential flux")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    save(fig, "gp1d_ajuste_ejemplo")
    plt.close(fig)


def gp2d_fit(df, oid=252646):
    light_curve = df[df.object_id == oid]
    estimator_model = gp2d.fit_gp2d(light_curve)
    rasgos = gp2d.object_gp2d_features(light_curve)
    t0, tt = estimator_model["t0"], estimator_model["tt"]
    tgrid = np.linspace(0, tt.max(), 160)
    ondas = np.array([gp2d.WAVELENGTHS[i] for i in range(6)])
    wgrid = np.linspace(ondas.min(), ondas.max(), 120)
    T, W = np.meshgrid(tgrid, wgrid)
    mu = (
        estimator_model["gp"]
        .predict(
            estimator_model["fn"],
            np.column_stack([T.ravel(), W.ravel()]),
            return_cov=False,
            return_var=False,
        )
        .reshape(T.shape)
        * estimator_model["scale"]
    )

    colores = {
        "u": "violet",
        "g": "green",
        "r": "red",
        "i": "darkred",
        "z": "saddlebrown",
        "y": "black",
    }
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.5))
    im = ax[0].pcolormesh(tgrid, wgrid, mu, cmap="viridis", shading="auto")
    for i in range(6):
        s = light_curve[light_curve.passband == i]
        ax[0].scatter(
            s.mjd - t0,
            [gp2d.WAVELENGTHS[i]] * len(s),
            s=13,
            color="white",
            edgecolor="k",
            lw=0.3,
            alpha=0.7,
        )
        ax[0].text(tt.max() * 1.01, gp2d.WAVELENGTHS[i], "ugrizy"[i], va="center", fontsize=8)
    ax[0].axvline(rasgos["gp2d_t_peak"], color="w", ls="--", lw=1)
    ax[0].set(
        xlabel="Days since first observation",
        ylabel="Wavelength (Å)",
        title="Posterior mean over time and wavelength",
    )
    fig.colorbar(im, ax=ax[0], label="Differential flux")

    tline = np.linspace(0, tt.max(), 200)
    for i, band_index in enumerate("ugrizy"):
        ax[1].plot(
            tline,
            gp2d._predict(estimator_model, tline + t0, band_index),
            color=colores[band_index],
            lw=1.5,
            label=band_index,
        )
        s = light_curve[light_curve.passband == i]
        ax[1].errorbar(
            s.mjd - t0,
            s.flux,
            yerr=s.flux_err,
            fmt="o",
            ms=3,
            color=colores[band_index],
            alpha=0.45,
            lw=0.6,
        )
    ax[1].axvline(rasgos["gp2d_t_peak"], color="gray", ls="--", lw=1, label="Global peak")
    ax[1].set(
        xlabel="Days since first observation",
        ylabel="Differential flux",
        title="GP2D sections at the six passbands",
    )
    ax[1].legend(ncol=3, fontsize=7)
    fig.suptitle(f"Object {oid}: GP2D fit", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "gp2d_superficie_ejemplo")
    plt.close(fig)


if __name__ == "__main__":
    curvas = pd.read_csv(config.TRAIN_CLEAN_CSV)
    object_metadata = pd.read_csv(config.METADATA_CLEAN_CSV)[["object_id", "target"]]
    example_curves(curvas, object_metadata)
    gp1d_fit(curvas)
    gp2d_fit(curvas)
    print("Figures vectoriales of the chapter 3 regenerated.")
