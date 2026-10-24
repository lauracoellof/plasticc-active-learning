"""Check pinned versions, compiled imports and optional Conda file integrity."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = {"python", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "george", "pytest"}
IMPORTS = ("numpy", "pandas", "scipy.linalg", "sklearn.ensemble", "joblib", "george")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Check every environment.yml dependency, including notebook and SHAP tools",
    )
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="Compare core compiled files with installed Conda package records",
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Check metadata and optional hashes only; this does not verify runtime execution",
    )
    args = parser.parse_args()
    errors = []
    print(f"Python executable: {sys.executable}", flush=True)
    pins = re.findall(
        r"^\s+-\s+([\w-]+)=([\w.]+)", (ROOT / "environment.yml").read_text(), re.MULTILINE
    )
    for name, expected in pins:
        if not args.full and name not in CORE:
            continue
        try:
            actual = platform.python_version() if name == "python" else metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        print(f"{name}: {actual} (required {expected})", flush=True)
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    for name in (() if args.skip_imports else IMPORTS):
        try:
            importlib.import_module(name)
            print(f"Import OK: {name}", flush=True)
        except Exception as exc:
            message = str(exc)
            original = re.search(r"Original error was:\s*([^\n]+)", message)
            message = original.group(1) if original else " ".join(message.split())[:400]
            errors.append(f"{name}: {type(exc).__name__}: {message}")
            print(f"Import FAILED: {name}: {message}", flush=True)
    if args.integrity:
        records = Path(sys.prefix) / "conda-meta"
        checked = mismatches = 0
        if not records.is_dir():
            errors.append("Conda integrity checks require a Conda environment")
        else:
            for path in sorted(records.glob("*.json")):
                package = json.loads(path.read_text(encoding="utf-8"))
                if package.get("name") not in CORE:
                    continue
                for entry in package.get("paths_data", {}).get("paths", []):
                    relative = entry["_path"]
                    expected = entry.get("sha256_in_prefix", entry.get("sha256"))
                    if not relative.endswith((".pyd", ".dll", ".so")) or not expected:
                        continue
                    binary = Path(sys.prefix) / relative
                    checked += 1
                    if (
                        not binary.is_file()
                        or hashlib.sha256(binary.read_bytes()).hexdigest() != expected
                    ):
                        mismatches += 1
                        if mismatches <= 10:
                            print(f"Integrity mismatch: {relative}", flush=True)
            print(f"Compiled files checked: {checked}; mismatches: {mismatches}", flush=True)
            if not checked:
                errors.append("No compiled-file hashes were available; integrity was not verified")
            if mismatches:
                errors.append(f"{mismatches} compiled files differ from their Conda records")
    for error in errors:
        print(f"ERROR: {error}", flush=True)
    if errors:
        print(
            "Environment check failed. Resolve the reported version, integrity or runtime-loading errors before regenerating scientific results."
        )
    elif args.skip_imports:
        print(
            "Static environment checks passed. Runtime imports were skipped and remain unverified."
        )
    else:
        print("Environment check passed. Run python -m pytest -q next.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
