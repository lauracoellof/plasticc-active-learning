"""Batch orchestration must forward the configured paths to feature extraction."""

import pandas as pd
import pytest

from src import pipeline


def test_batch_pipeline_forwards_paths(monkeypatch):
    """Catch keyword/API drift before any expensive feature extraction or fitting."""
    paths = pipeline.select_batch(4)
    metadata = pd.DataFrame({"object_id": [1, 2], "true_target": [6, 15]})
    monkeypatch.setattr(pipeline, "select_batch", lambda batch: paths)
    monkeypatch.setattr(pipeline, "_load_batch", lambda *args: (None, metadata))

    class FeaturesReached(Exception):
        pass

    def features(curves, table, *, batch_paths, usar_cache, guardar_cache, verbose):
        assert curves is None
        assert table is metadata
        assert batch_paths is paths
        assert usar_cache is True
        raise FeaturesReached

    monkeypatch.setattr(pipeline, "features_batch", features)
    with pytest.raises(FeaturesReached):
        pipeline.process_batch_al(batch=4, verbose=False)
