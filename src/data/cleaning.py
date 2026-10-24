"""Detection, signal-to-noise and observation-count cuts for PLAsTiCC light curves."""


def cleaning_pipeline(df, snr_min=3, min_obs=5, verbose=True):
    """Apply detected==1, strict absolute-SNR and per-object observation-count cuts.

    Count observations across bands, retain negative detected fluxes when their
    absolute SNR passes, and return rows ordered by object_id and MJD. The input
    requires object_id, mjd, detected, flux and flux_err columns."""
    log = []

    n = len(df)
    df = df[df["detected"] == 1].copy()
    log.append(("1 · detected = 1", n - len(df), len(df)))

    n = len(df)
    df = df[df["flux"].abs() / df["flux_err"] > snr_min].copy()
    log.append((f"2 · |SNR| > {snr_min}", n - len(df), len(df)))

    n = len(df)
    conteo = df.groupby("object_id")["flux"].transform("size")
    df = df[conteo >= min_obs].copy()
    log.append((f"3 · obs/object >= {min_obs}", n - len(df), len(df)))

    df = df.sort_values(["object_id", "mjd"]).reset_index(drop=True)

    if verbose:
        print(f"{'Step':<24}{'Removed':>12}{'Remaining':>12}")
        for paso, elim, rest in log:
            print(f"{paso:<24}{elim:>12,}{rest:>12,}")
        print(f"\nRemaining unique objects: {df['object_id'].nunique():,}")

    return df


def synchronize_metadata(metadata, df_clean):
    """Retain metadata rows whose object_id survives light-curve cleaning."""
    object_ids = df_clean["object_id"].unique()
    return metadata[metadata["object_id"].isin(object_ids)].copy()
