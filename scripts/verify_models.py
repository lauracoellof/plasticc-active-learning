"Verify models."

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.models.classification import load_e2gp_artifact


def sha256(file_path: Path, tamano_bloque: int = 8 * 1024 * 1024) -> str:
    summary_table = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for feature_block_name in iter(lambda: file_handle.read(tamano_bloque), b""):
            summary_table.update(feature_block_name)
    return summary_table.hexdigest()


def verify() -> None:
    input_entries = {}
    for text_line in config.CHECKSUMS_MODELOS.read_text(encoding="utf-8").splitlines():
        summary_table, item_name = text_line.split(maxsplit=1)
        input_entries[item_name] = summary_table
    expected_entries = {config.E2GP_MODEL.name, config.E2GP_METADATA.name}
    if set(input_entries) != expected_entries:
        raise SystemExit(f"Unexpected model-checksum entries: {sorted(input_entries)}")

    for item_name, expected_value_value in input_entries.items():
        file_path = config.MODEL_DIR / item_name
        actual_value = sha256(file_path)
        if actual_value != expected_value_value:
            raise SystemExit(f"Checksum mismatch for {item_name}")
        print(f"OK  {item_name}")

    estimator_model, feature_columns, object_metadata = load_e2gp_artifact()
    if len(feature_columns) != object_metadata[
        "n_features"
    ] or estimator_model.n_features_in_ != len(feature_columns):
        raise SystemExit("Inconsistent feature contract")
    print(
        f"Model loaded: {object_metadata['nombre']} · "
        f"{object_metadata['n_train']} objects · {len(feature_columns)} features"
    )


if __name__ == "__main__":
    verify()
