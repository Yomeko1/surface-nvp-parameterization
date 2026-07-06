from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from surface_nvp.geometry.boundary import boundary_mask, extract_boundary_loop
from surface_nvp.geometry.topology import build_vertex_neighbors
from surface_nvp.injectivity.signed_area import triangle_signed_areas

from .boundary_map import map_boundary_to_circle, map_boundary_to_square


def tutte_parameterize(vertices: np.ndarray, faces: np.ndarray, boundary_mode: str = "circle") -> np.ndarray:
    num_vertices = int(vertices.shape[0])
    loop = extract_boundary_loop(faces)
    is_boundary = boundary_mask(num_vertices, loop)
    interior = np.where(~is_boundary)[0]
    if interior.size == 0:
        raise ValueError("mesh has no interior vertices")
    uv = np.zeros((num_vertices, 2), dtype=np.float64)
    if boundary_mode == "circle":
        boundary_uv = map_boundary_to_circle(vertices, loop)
    elif boundary_mode == "square":
        boundary_uv = map_boundary_to_square(vertices, loop)
    else:
        raise ValueError(f"unknown boundary mode: {boundary_mode}")
    uv[np.asarray(loop, dtype=np.int64)] = boundary_uv

    idx_of = {int(v): i for i, v in enumerate(interior)}
    neighbors = build_vertex_neighbors(faces, num_vertices)
    mat = lil_matrix((interior.size, interior.size), dtype=np.float64)
    rhs = np.zeros((interior.size, 2), dtype=np.float64)

    for row, vi in enumerate(interior):
        nbrs = neighbors[int(vi)]
        if not nbrs:
            raise ValueError("interior vertex has no neighbors")
        w = 1.0 / float(len(nbrs))
        mat[row, row] = 1.0
        for vj in nbrs:
            if is_boundary[vj]:
                rhs[row] += w * uv[vj]
            else:
                mat[row, idx_of[int(vj)]] -= w

    solved = spsolve(mat.tocsr(), rhs)
    uv[interior] = solved
    if float(triangle_signed_areas(uv, faces).sum()) < 0.0:
        uv[:, 1] *= -1.0
    return uv
