"Generate fine allocation figure."

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "data" / "processed" / "dos_canales_batch99_fino_10s.csv"
SALIDA = ROOT / "reports" / "figures" / "apendice_reparto_alpha_fino_10s.png"


def main() -> None:
    data = pd.read_csv(ENTRADA)
    finales = data.sort_values("iter").groupby(["frac_desc", "semilla"], as_index=False).tail(1)
    n_semillas = finales.groupby("frac_desc")["semilla"].nunique()
    if not (n_semillas == 10).all():
        raise ValueError(f"Incomplete allocation sweep: {n_semillas.to_dict()}")

    summary_table = finales.groupby("frac_desc").agg(
        MF1=("MF1", "mean"),
        MF1_ee=("MF1", lambda x: x.std(ddof=1) / np.sqrt(x.size)),
        hallazgos=("n_descubiertas", "mean"),
        hallazgos_ee=("n_descubiertas", lambda x: x.std(ddof=1) / np.sqrt(x.size)),
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    color = "#2F6690"
    band_index = "#88B8A9"
    fig, plot_axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)

    x = summary_table.index.to_numpy(dtype=float)
    plot_axes[0].errorbar(
        x,
        summary_table["hallazgos"],
        yerr=summary_table["hallazgos_ee"],
        marker="o",
        linewidth=2,
        capsize=3,
        color=color,
    )
    plot_axes[0].axvspan(0.3, 0.6, color=band_index, alpha=0.20)
    plot_axes[0].set(
        xlabel="Discovery allocation fraction, $\\alpha$",
        ylabel="Discovered 99x anomalies",
        title="Discovery",
    )

    plot_axes[1].errorbar(
        x,
        summary_table["MF1"],
        yerr=summary_table["MF1_ee"],
        marker="o",
        linewidth=2,
        capsize=3,
        color=color,
    )
    plot_axes[1].axvspan(
        0.3, 0.6, color=band_index, alpha=0.20, label="operating range $0.3$--$0.6$"
    )
    plot_axes[1].set(
        xlabel="Discovery allocation fraction, $\\alpha$",
        ylabel="Macro-F1",
        title="Classification",
    )
    plot_axes[1].legend(loc="lower left")

    for eje in plot_axes:
        eje.set_xticks(x)
        eje.grid(alpha=0.25)

    fig.suptitle("Sensitivity to budget allocation (10 seeds)", fontsize=12)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(SALIDA)


if __name__ == "__main__":
    main()
