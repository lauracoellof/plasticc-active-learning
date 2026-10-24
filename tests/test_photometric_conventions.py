"""Flux zero points and photometric proxy conventions."""

import numpy as np

from src.features import gp, gp2d


def test_gp_models_use_the_same_decline_sign():
    expected_value_value = -2.5 * np.log10(50.0 / 100.0)
    assert np.isclose(gp._delta_mag(100.0, 50.0, 1e-3), expected_value_value)
    assert np.isclose(gp2d._delta_mag(100.0, 50.0, 1e-3), expected_value_value)
    assert expected_value_value > 0.0


def test_gp1d_rejects_phases_outside_observed_support():
    fit_result = {"t0": 100.0, "X": np.array([[0.0], [10.0], [20.0]])}
    assert gp._within_support(fit_result, 100.0)
    assert gp._within_support(fit_result, 120.0)
    assert not gp._within_support(fit_result, 120.01)
