"Verify raw data."

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "checksums.sha256"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def required_files() -> set[str]:
    """Return the fourteen raw source filenames required by the project."""
    return {
        config.TRAIN_CSV.name,
        config.METADATA_CSV.name,
        config.BatchConfig(1).test_metadata_csv.name,
        *(config.BatchConfig(batch).test_lc.name for batch in range(1, 12)),
    }


def read_manifest(file_path: Path = MANIFEST) -> dict[str, str]:
    """Read distinct filenames and valid SHA-256 digests from a checksum manifest."""
    recorded_rows: dict[str, str] = {}
    for number_value, text_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        if not text_line.strip() or text_line.lstrip().startswith("#"):
            continue
        partes = text_line.split(maxsplit=1)
        if len(partes) != 2:
            raise ValueError(f"Line {number_value} invalid in {file_path.name}")
        summary_table, item_name = partes
        if len(summary_table) != 64 or any(c not in "0123456789abcdef" for c in summary_table):
            raise ValueError(f"Invalid SHA-256 for {item_name!r}")
        if item_name in recorded_rows:
            raise ValueError(f"Duplicate manifest entry: {item_name}")
        recorded_rows[item_name] = summary_table
    return recorded_rows


def sha256(file_path: Path, tamano_bloque: int = 8 * 1024 * 1024) -> str:
    "Sha256."
    summary_table = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for feature_block_name in iter(lambda: file_handle.read(tamano_bloque), b""):
            summary_table.update(feature_block_name)
    return summary_table.hexdigest()


def verify() -> None:
    "Verify."
    manifiesto = read_manifest()
    requeridos = required_files()
    if set(manifiesto) != requeridos:
        missing_entries = sorted(requeridos - set(manifiesto))
        extra_entries = sorted(set(manifiesto) - requeridos)
        raise SystemExit(
            f"Manifest desincronizado; missing={missing_entries}, sobran={extra_entries}"
        )

    ausentes = sorted(item_name for item_name in requeridos if not (RAW / item_name).is_file())
    if ausentes:
        raise SystemExit(f"Missing files in {RAW}: {ausentes}")

    for indice, (item_name, expected_value_value) in enumerate(manifiesto.items(), 1):
        actual_value = sha256(RAW / item_name)
        if actual_value != expected_value_value:
            raise SystemExit(
                f"Integridad incorrect for {item_name}: expected={expected_value_value}, actual={actual_value}"
            )
        print(f"[{indice:02d}/{len(manifiesto):02d}] OK  {item_name}", flush=True)

    print(
        f"Integridad verificada: {len(manifiesto)}/{len(manifiesto)} "
        f"files match with {MANIFEST.name}"
    )


if __name__ == "__main__":
    verify()
