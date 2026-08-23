from pathlib import Path

import surface_nvp


def test_package_and_project_versions_match_v24():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert surface_nvp.__version__ == "2.4.0"
    assert 'version = "2.4.0"' in pyproject
