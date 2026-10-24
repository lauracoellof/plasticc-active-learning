"""Feature-cache integrity, ordered schemas and extractor compatibility checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


CACHE_FORMAT_VERSION = 1
REPRESENTATION_VERSION = "e2gp-2.0.0"

_SOURCES = {
    "estadisticas": ("statistics.py",),
    "gp1d": ("gp.py",),
    "gp2d": ("gp2d.py",),
    "e2gp": ("statistics.py", "gp.py", "gp2d.py"),
}


class InvalidFeatureCache(RuntimeError):
    """A feature table is missing, incompatible or inconsistent with its manifest."""


def manifest_path(ruta_csv):
    """Return the JSON sidecar path associated with a feature CSV."""
    ruta_csv = Path(ruta_csv)
    return ruta_csv.with_name(ruta_csv.name + ".meta.json")


def _file_sha256(file_path):
    h = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for feature_block_name in iter(lambda: f.read(1024 * 1024), b""):
            h.update(feature_block_name)
    return h.hexdigest()


def _source_bytes(file_path):
    """Read Python source with canonical LF endings, as Python and Git interpret it."""
    return Path(file_path).read_bytes().replace(b"\r\n", b"\n")


def _source_sha256(file_path):
    return hashlib.sha256(_source_bytes(file_path)).hexdigest()


def extractor_fingerprint(feature_family_name):
    """Hash extractor sources independently of Windows/Unix line-ending conventions."""
    try:
        display_names = _SOURCES[feature_family_name]
    except KeyError as exc:
        raise ValueError(f"Unknown cache family: {feature_family_name}") from exc
    base = Path(__file__).resolve().parent
    h = hashlib.sha256(REPRESENTATION_VERSION.encode("utf-8"))
    for item_name in display_names:
        h.update(item_name.encode("utf-8"))
        h.update(_source_bytes(base / item_name))
    return h.hexdigest()


def write_cache(result_table, ruta_csv, feature_family_name, *, producer_name, extra_data=None):
    """Write a feature CSV and its ordered-schema, extractor and content-hash manifest."""
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(ruta_csv, index=False)
    feature_columns = list(result_table.columns)
    esquema = json.dumps(feature_columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    manifiesto = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "familia": feature_family_name,
        "productor": producer_name,
        "n_filas": int(len(result_table)),
        "n_columnas": len(feature_columns),
        "columnas": feature_columns,
        "feature_schema_sha256": hashlib.sha256(esquema).hexdigest(),
        "extractor_sha256": extractor_fingerprint(feature_family_name),
        "csv_sha256": _file_sha256(ruta_csv),
    }
    if extra_data:
        manifiesto["extra"] = extra_data
    manifest_path(ruta_csv).write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifiesto


def register_existing_cache(ruta_csv, feature_family_name, *, producer_name, extra_data=None):
    """Record schema and integrity metadata for an existing CSV without changing its bytes."""
    ruta_csv = Path(ruta_csv)
    feature_columns = list(pd.read_csv(ruta_csv, nrows=0).columns)
    row_count = sum(
        len(b) for b in pd.read_csv(ruta_csv, usecols=[feature_columns[0]], chunksize=500_000)
    )
    esquema = json.dumps(feature_columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    manifiesto = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "familia": feature_family_name,
        "productor": producer_name,
        "n_filas": int(row_count),
        "n_columnas": len(feature_columns),
        "columnas": feature_columns,
        "feature_schema_sha256": hashlib.sha256(esquema).hexdigest(),
        "extractor_sha256": extractor_fingerprint(feature_family_name),
        "csv_sha256": _file_sha256(ruta_csv),
    }
    if extra_data:
        manifiesto["extra"] = extra_data
    manifest_path(ruta_csv).write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifiesto


def validate_cache(ruta_csv, feature_family_name, *, verify_csv=False):
    """Check the representation, extractor and column order; optionally verify all CSV bytes.

    Historical fingerprints are accepted only by the explicit compatibility record
    and only while every reviewed extractor source still matches its recorded hash."""
    ruta_csv = Path(ruta_csv)
    ruta_meta = manifest_path(ruta_csv)
    if not ruta_csv.exists() or not ruta_meta.exists():
        raise InvalidFeatureCache(f"Missing cache or manifest: {ruta_csv.name}")
    metadata_table = json.loads(ruta_meta.read_text(encoding="utf-8"))
    expected_value_value = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "familia": feature_family_name,
        "extractor_sha256": extractor_fingerprint(feature_family_name),
    }
    compatible_fingerprints = set()
    bridge_path = Path(__file__).resolve().parents[2] / "feature_cache_compatibility.json"
    if bridge_path.exists():
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        current_sources = bridge.get("source_sha256", {})
        if current_sources and all(
            _source_sha256(Path(__file__).parent / name) == digest
            for name, digest in current_sources.items()
        ):
            legacy = bridge.get("legacy_fingerprints", {}).get(feature_family_name)
            if legacy is not None:
                compatible_fingerprints.add(legacy)
            compatible_fingerprints.update(
                bridge.get("reviewed_fingerprints", {}).get(feature_family_name, [])
            )
    error_messages = [
        f"{k}={metadata_table.get(k)!r} (expected {v!r})"
        for k, v in expected_value_value.items()
        if metadata_table.get(k) != v
        and not (k == "extractor_sha256" and metadata_table.get(k) in compatible_fingerprints)
    ]
    cabecera = list(pd.read_csv(ruta_csv, nrows=0).columns)
    if cabecera != metadata_table.get("columnas"):
        error_messages.append("The CSV schema does not match its manifest")
    if verify_csv and _file_sha256(ruta_csv) != metadata_table.get("csv_sha256"):
        error_messages.append("The CSV hash does not match its manifest")
    if error_messages:
        raise InvalidFeatureCache(f"Outdated cache {ruta_csv.name}: " + "; ".join(error_messages))
    return metadata_table


def load_cache(ruta_csv, feature_family_name, *, verify_csv=True):
    """Validate a feature CSV before reading it into a pandas DataFrame."""
    validate_cache(ruta_csv, feature_family_name, verify_csv=verify_csv)
    return pd.read_csv(ruta_csv)
