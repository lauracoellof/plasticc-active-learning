"""Agreement between the declared Conda and pip scientific package versions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _conda_dependencies() -> dict[str, str]:
    dependencias: dict[str, str] = {}
    for text_line in (ROOT / "environment.yml").read_text(encoding="utf-8").splitlines():
        content_text = text_line.strip()
        if not content_text.startswith("- ") or "=" not in content_text:
            continue
        item_name, version = content_text[2:].split("=", 1)
        dependencias[item_name.strip()] = version.split("#", 1)[0].strip()
    return dependencias


def _pip_dependencies() -> dict[str, str]:
    dependencias: dict[str, str] = {}
    for text_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        content_text = text_line.strip()
        if not content_text or content_text.startswith("#"):
            continue
        assert "==" in content_text, f"Unpinned pip dependency: {content_text}"
        item_name, version = content_text.split("==", 1)
        dependencias[item_name.strip()] = version.strip()
    return dependencias


def test_conda_and_pip_versions_agree():
    "Verify conda and pip versions agree."
    conda = _conda_dependencies()
    pip = _pip_dependencies()

    assert conda.pop("python") == "3.11.15"
    assert conda == pip
