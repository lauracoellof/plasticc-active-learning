"""Published model files must retain their recorded bytes after Git checkout."""

import hashlib
from pathlib import Path


def test_published_model_checksums_match_artifact_bytes():
    directory = Path(__file__).resolve().parents[1] / "models"
    entries = {}
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    assert set(entries) == {"e2gp_150_calibrado.joblib", "e2gp_150_calibrado.meta.json"}
    for name, digest in entries.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
