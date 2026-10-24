"""Portable batch paths and object-level subsampling."""

from dataclasses import FrozenInstanceError

import pytest

import config
from src import pipeline


def _legacy_global_state():
    return (
        config.TEST_BATCH,
        config.TEST_LC,
        config.TEST_METADATA_CLEAN_CSV,
        config.FEATURES_TEST_CSV,
        config.FEATURES_GP_TEST_CSV,
        config.FEATURES_GP2D_TEST_CSV,
    )


def test_batch_paths_do_not_collide():
    b1 = config.BatchConfig(1)
    b4 = config.BatchConfig(4)
    pool = config.BatchConfig(99)

    assert b1.features_test_csv.name == "features_estadisticas_test.csv"
    assert b4.features_test_csv.name == "features_estadisticas_test_b4.csv"
    assert pool.features_test_csv.name == "features_estadisticas_test_b99.csv"
    assert b4.test_lc.name == "plasticc_test_lightcurves_04.csv.gz"
    assert (
        len({b1.test_metadata_clean_csv, b4.test_metadata_clean_csv, pool.test_metadata_clean_csv})
        == 3
    )


def test_batch_selection_does_not_mutate_global_configuration():
    antes = _legacy_global_state()
    rutas_1 = pipeline.select_batch(1)
    rutas_6 = pipeline.select_batch(6)
    despues = _legacy_global_state()

    assert antes == despues
    assert rutas_1.batch == 1
    assert rutas_6.batch == 6
    assert rutas_1.features_test_csv != rutas_6.features_test_csv


def test_batch_configuration_is_immutable_and_validates_its_number():
    batch_paths = config.BatchConfig(4)
    with pytest.raises(FrozenInstanceError):
        batch_paths.batch = 5
    with pytest.raises(ValueError, match="batch must be"):
        config.BatchConfig(12)


def test_legacy_aliases_refer_only_to_the_default_batch():
    batch_paths = config.BatchConfig(config.TEST_BATCH)
    assert config.TEST_LC == batch_paths.test_lc
    assert config.TEST_METADATA_CLEAN_CSV == batch_paths.test_metadata_clean_csv
    assert config.FEATURES_TEST_CSV == batch_paths.features_test_csv
    assert config.FEATURES_GP_TEST_CSV == batch_paths.features_gp_test_csv
    assert config.FEATURES_GP2D_TEST_CSV == batch_paths.features_gp2d_test_csv
