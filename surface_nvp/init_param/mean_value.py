from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from surface_nvp.geometry.boundary import boundary_mask, extract_boundary_loop
from surface_nvp.geometry.topology import build_vertex_neighbors
from surface_nvp.injectivity.signed_area import triangle_signed_areas

from .boundary_map import map_boundary_to_circle, map_boundary_to_square


def mean_value_parameterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_mode: str = "circle",
) -> np.ndarray:
    """Compute a fixed-convex-boundary map using positive mean-value weights."""
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

    weights = _mean_value_weights(vertices, faces)
    neighbors = build_vertex_neighbors(faces, num_vertices)
    idx_of = {int(v): i for i, v in enumerate(interior)}
    mat = lil_matrix((interior.size, interior.size), dtype=np.float64)
    rhs = np.zeros((interior.size, 2), dtype=np.float64)

    for row, vi_value in enumerate(interior):
        vi = int(vi_value)
        nbrs = neighbors[vi]
        if not nbrs:
            raise ValueError("interior vertex has no neighbors")
        row_weights = np.asarray([weights[vi].get(int(vj), 0.0) for vj in nbrs])
        total = float(row_weights.sum())
        if not np.isfinite(total) or total <= 0.0 or np.any(row_weights <= 0.0):
            raise ValueError("failed to construct positive mean-value weights")
        row_weights /= total
        mat[row, row] = 1.0
        for vj, weight in zip(nbrs, row_weights):
            if is_boundary[vj]:
                rhs[row] += float(weight) * uv[vj]
            else:
                mat[row, idx_of[int(vj)]] -= float(weight)

    solved = spsolve(mat.tocsr(), rhs)
    if not np.all(np.isfinite(solved)):
        raise ValueError("mean-value linear solve produced non-finite coordinates")
    uv[interior] = solved
    if float(triangle_signed_areas(uv, faces).sum()) < 0.0:
        uv[:, 1] *= -1.0
    return uv


def _mean_value_weights(vertices: np.ndarray, faces: np.ndarray) -> list[dict[int, float]]:
    """Return Floater mean-value weights before per-row normalization."""
    weights: list[dict[int, float]] = [dict() for _ in range(len(vertices))]
    for face in faces:
        ids = [int(value) for value in face]
        points = vertices[ids]
        for local_i in range(3):
            vi = ids[local_i]
            vj = ids[(local_i + 1) % 3]
            vk = ids[(local_i + 2) % 3]
            edge_j = points[(local_i + 1) % 3] - points[local_i]
            edge_k = points[(local_i + 2) % 3] - points[local_i]
            len_j = float(np.linalg.norm(edge_j))
            len_k = float(np.linalg.norm(edge_k))
            cross_norm = float(np.linalg.norm(np.cross(edge_j, edge_k)))
            dot = float(np.dot(edge_j, edge_k))
            if min(len_j, len_k, cross_norm) <= 1e-15:
                raise ValueError("mean-value initialization requires non-degenerate triangles")
            angle = float(np.arctan2(cross_norm, dot))
            tan_half = float(np.tan(0.5 * angle))
            if not np.isfinite(tan_half) or tan_half <= 0.0:
                raise ValueError("failed to compute a positive triangle half-angle")
            weights[vi][vj] = weights[vi].get(vj, 0.0) + tan_half / len_j
            weights[vi][vk] = weights[vi].get(vk, 0.0) + tan_half / len_k
    return weights
