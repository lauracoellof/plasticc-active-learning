"""Per-band statistics, cross-band descriptors and derived feature conventions."""

import numpy as np
import pandas as pd
from src.features.statistics import (
    FLUX_ZEROPOINT,
    add_derived_features,
    build_statistics,
)


def test_statistical_feature_shape():
    df = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 1],
            "passband": [2, 2, 3, 3],
            "flux": [10.0, 12.0, 5.0, 7.0],
        }
    )
    out = build_statistics(df)
    assert out.shape[0] == 1
    assert "mean_r" in out.columns and "mean_i" in out.columns


def test_peak_magnitude_uses_the_zero_point_and_distance():
    result_table = pd.DataFrame(
        {
            "gp_peak_flux_r": [100.0, 100.0, 0.0],
            "distmod": [40.0, 0.0, 40.0],
            "hostgal_photoz": [0.2, 0.0, 0.2],
        }
    )
    out = add_derived_features(result_table, band_names=("r",))
    expected_value_value = FLUX_ZEROPOINT - 2.5 * np.log10(100.0) - 40.0
    assert np.isclose(out.loc[0, "gp_peak_absmag_r"], expected_value_value)
    assert out.loc[1, "gp_peak_absmag_r"] == 0.0
    assert out.loc[2, "gp_peak_absmag_r"] == 0.0
