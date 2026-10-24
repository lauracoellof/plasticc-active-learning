"Extract pool anomaly curves."

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src.data.cleaning import cleaning_pipeline
from src.features import extra_data as ex


def pool_anomaly_ids():
    "Pool anomaly ids."
    m = pd.read_csv(config.DIR_PROCESSED / "test_batch99_metadata_clean.csv")
    return set(m.loc[m["true_target"] >= 990, "object_id"])


def batch_anomaly_curves(batch, object_ids, chunksize=2_000_000):
    "Batch anomaly curves."
    path = config.DIR_RAW / f"plasticc_test_lightcurves_{batch:02d}.csv.gz"
    chunks_list = []
    for ch in pd.read_csv(path, chunksize=chunksize):
        ch = ch.rename(columns={"detected_bool": "detected"})
        sub = ch[ch["object_id"].isin(object_ids)]
        if len(sub):
            chunks_list.append(sub)
    return pd.concat(chunks_list, ignore_index=True) if chunks_list else pd.DataFrame()


def main(batches=None):
    batches = batches or list(range(1, 12))
    object_ids = pool_anomaly_ids()
    metadata_table = pd.read_csv(
        config.TEST_METADATA_CSV, usecols=["object_id", "true_target", "distmod", "hostgal_photoz"]
    )
    metadata_table = metadata_table[metadata_table["object_id"].isin(object_ids)].copy()

    curvas = []
    for b in batches:
        t = time.time()
        c = batch_anomaly_curves(b, object_ids)
        n = 0
        if len(c):
            c = cleaning_pipeline(c, snr_min=config.SNR_MIN, min_obs=config.MIN_OBS, verbose=False)
            n = c["object_id"].nunique()
            curvas.append(c)
        print(f"batch {b:2d}: {n:4d} anomalies limpias · {time.time() - t:.0f}s", flush=True)

    lc = pd.concat(curvas, ignore_index=True)
    metadata_table = metadata_table[metadata_table["object_id"].isin(lc["object_id"].unique())]
    result_table = ex.build_extra_feature_table(
        lc, metadata_table=metadata_table, progress_interval=200
    )
    out = config.DIR_PROCESSED / "features_extra_test_b99.csv"
    result_table.to_csv(out, index=False)
    print(f"saved {len(result_table)} anomalies -> {out}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]] or None
    main(args)
