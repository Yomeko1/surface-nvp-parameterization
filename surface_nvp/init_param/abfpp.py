from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from surface_nvp.io.mesh_data import MeshData
from surface_nvp.io.obj_io import load_obj, save_obj


def abfpp_parameterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    executable: str | Path | None,
) -> np.ndarray:
    """Run the optional OpenABF command-line wrapper and return one UV per vertex."""
    if executable is None:
        raise ValueError("ABF++ initialization requires abfpp_executable")
    executable_path = Path(executable).expanduser().resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(f"ABF++ executable not found: {executable_path}")

    mesh = MeshData(vertices, faces)
    with tempfile.TemporaryDirectory(prefix="surface_nvp_abfpp_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "input.obj"
        output_path = temp_root / "output.obj"
        save_obj(input_path, mesh)
        completed = subprocess.run(
            [str(executable_path), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"ABF++ runner failed with exit code {completed.returncode}: {details}")
        if not output_path.is_file():
            raise RuntimeError("ABF++ runner did not create its output OBJ")
        result = load_obj(output_path)

    if result.uv is None:
        raise RuntimeError("ABF++ output contains no UV coordinates")
    if result.vertices.shape != vertices.shape:
        raise RuntimeError("ABF++ output has a different vertex count")
    if result.faces.shape != faces.shape or not np.array_equal(result.faces, faces):
        raise RuntimeError("ABF++ output has different triangle topology")
    return result.uv.copy()
