from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from surface_nvp.io import load_mesh
from surface_nvp.io.mesh_data import MeshData
from surface_nvp.losses.distortion import jacobian_determinants

from .tutte import tutte_parameterize


def resolve_initial_uv(
    mesh: MeshData,
    method: str = "tutte",
    boundary_mode: str = "circle",
    initial_uv_path: str | Path | None = None,
    prim_path: str | None = None,
    geometry_scale: bool = False,
) -> np.ndarray:
    if initial_uv_path is not None:
        initial_mesh = load_mesh(initial_uv_path, prim_path=prim_path)
        if initial_mesh.uv is None:
            raise ValueError(f"initial UV mesh has no UV coordinates: {initial_uv_path}")
        if initial_mesh.vertices.shape[0] != mesh.vertices.shape[0]:
            raise ValueError("initial UV mesh must have the same number of vertices as the input mesh")
        if initial_mesh.faces.shape != mesh.faces.shape or not np.array_equal(initial_mesh.faces, mesh.faces):
            raise ValueError("initial UV mesh must have the same triangle topology as the input mesh")
        uv = initial_mesh.uv.copy()
    elif mesh.uv is not None:
        uv = mesh.uv.copy()
    elif method == "tutte":
        uv = tutte_parameterize(mesh.vertices, mesh.faces, boundary_mode=boundary_mode)
    else:
        raise ValueError(f"unsupported init method: {method}")
    if geometry_scale:
        uv = normalize_uv_geometry_scale(mesh.vertices, mesh.faces, uv)
    return uv


def normalize_uv_geometry_scale(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        determinants = jacobian_determinants(
            torch.as_tensor(vertices, dtype=torch.float64),
            torch.as_tensor(faces, dtype=torch.long),
            torch.as_tensor(uv, dtype=torch.float64),
        ).abs()
    finite_positive = determinants[torch.isfinite(determinants) & (determinants > 1e-12)]
    if finite_positive.numel() == 0:
        raise ValueError("cannot geometry-scale UV with no finite positive Jacobian determinants")
    scale = float(finite_positive.median().rsqrt())
    center = 0.5 * (uv.min(axis=0) + uv.max(axis=0))
    return (uv - center) * scale + center
