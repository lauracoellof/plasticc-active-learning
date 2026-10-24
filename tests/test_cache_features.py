"""Versioned feature-cache integrity and source-bound historical compatibility."""

import json

import pandas as pd
import pytest

from src.features.cache import (
    InvalidFeatureCache,
    load_cache,
    write_cache,
    manifest_path,
)


def test_versioned_cache_roundtrip(tmp_path):
    file_path = tmp_path / "features.csv"
    original = pd.DataFrame({"object_id": [1, 2], "mean_u": [3.0, 4.0]})
    write_cache(original, file_path, "estadisticas", producer_name="pytest")
    pd.testing.assert_frame_equal(load_cache(file_path, "estadisticas"), original)


def test_unversioned_or_incompatible_cache_is_rejected(tmp_path):
    file_path = tmp_path / "features.csv"
    pd.DataFrame({"object_id": [1]}).to_csv(file_path, index=False)
    with pytest.raises(InvalidFeatureCache):
        load_cache(file_path, "estadisticas")
    write_cache(pd.DataFrame({"object_id": [1]}), file_path, "estadisticas", producer_name="pytest")
    metadata_path = manifest_path(file_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["representation_version"] = "obsolete"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(InvalidFeatureCache):
        load_cache(file_path, "estadisticas")


def test_historical_cache_requires_exact_reviewed_extractor_sources(tmp_path, monkeypatch):
    """Editorial compatibility must never allow later extractor changes silently."""
    from pathlib import Path
    from src.features import cache

    path = tmp_path / "features.csv"
    write_cache(
        pd.DataFrame({"object_id": [1], "mean_u": [2.0]}),
        path,
        "estadisticas",
        producer_name="pytest",
    )
    compatibility = json.loads(
        (Path(cache.__file__).parents[2] / "feature_cache_compatibility.json").read_text()
    )
    metadata = json.loads(manifest_path(path).read_text())
    metadata["extractor_sha256"] = compatibility["legacy_fingerprints"]["estadisticas"]
    manifest_path(path).write_text(json.dumps(metadata))
    assert len(load_cache(path, "estadisticas")) == 1

    original_hash = cache._file_sha256
    monkeypatch.setattr(
        cache,
        "_file_sha256",
        lambda p: "modified-extractor" if Path(p).name == "statistics.py" else original_hash(p),
    )
    with pytest.raises(InvalidFeatureCache, match="extractor_sha256"):
        load_cache(path, "estadisticas")
