"""Required PLAsTiCC source-file inventory."""

from scripts.verify_raw_data import required_files, read_manifest


def test_raw_manifest_covers_exactly_the_configured_inputs():
    "Verify raw manifest covers exactly the configured inputs."
    manifiesto = read_manifest()

    assert set(manifiesto) == required_files()
    assert len(manifiesto) == 14
    assert len(set(manifiesto.values())) == 14
