from pathlib import Path

import surface_nvp


def test_package_and_project_versions_match_v30():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert surface_nvp.__version__ == "3.0.0"
    assert 'version = "3.0.0"' in pyproject
