"""Detection, signal-to-noise and observation-count cuts on synthetic light curves."""

import pandas as pd
from src.data.cleaning import cleaning_pipeline


def test_negative_differential_flux_is_retained():
    df = pd.DataFrame(
        {
            "object_id": [1] * 6,
            "mjd": [0, 1, 2, 3, 4, 5],
            "passband": [2] * 6,
            "flux": [-100.0, -90.0, 80.0, 70.0, -60.0, 50.0],
            "flux_err": [5.0] * 6,
            "detected": [1] * 6,
        }
    )
    out = cleaning_pipeline(df, snr_min=3, min_obs=5, verbose=False)

    assert (out["flux"] < 0).sum() == 3
