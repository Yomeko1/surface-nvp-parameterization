from __future__ import annotations

from pathlib import Path

import numpy as np

from surface_nvp.io import load_mesh
from surface_nvp.io.mesh_data import MeshData

from .tutte import tutte_parameterize


def resolve_initial_uv(
    mesh: MeshData,
    method: str = "tutte",
    boundary_mode: str = "circle",
    initial_uv_path: str | Path | None = None,
    prim_path: str | None = None,
) -> np.ndarray:
    if initial_uv_path is not None:
        initial_mesh = load_mesh(initial_uv_path, prim_path=prim_path)
        if initial_mesh.uv is None:
            raise ValueError(f"initial UV mesh has no UV coordinates: {initial_uv_path}")
        if initial_mesh.vertices.shape[0] != mesh.vertices.shape[0]:
            raise ValueError("initial UV mesh must have the same number of vertices as the input mesh")
        if initial_mesh.faces.shape != mesh.faces.shape or not np.array_equal(initial_mesh.faces, mesh.faces):
            raise ValueError("initial UV mesh must have the same triangle topology as the input mesh")
        return initial_mesh.uv.copy()

    if mesh.uv is not None:
        return mesh.uv.copy()
    if method == "tutte":
        return tutte_parameterize(mesh.vertices, mesh.faces, boundary_mode=boundary_mode)
    raise ValueError(f"unsupported init method: {method}")
