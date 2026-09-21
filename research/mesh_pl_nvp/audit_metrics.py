"""Independent, observation-only metrics for the frozen v3.1 experiment."""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import shapely
from shapely import STRtree
import torch

from surface_nvp.losses.distortion import symmetric_dirichlet_per_face
from surface_nvp.training.metrics import compute_distortion_metrics


def scales(vertices, initial_original_uv):
    result = {
        "D3D": float(np.linalg.norm(np.ptp(vertices, axis=0))),
        "LUV0": float(np.linalg.norm(np.ptp(initial_original_uv, axis=0))),
    }
    if not all(math.isfinite(v) and v > 0 for v in result.values()):
        raise ValueError("reference scales must be positive and finite")
    return result


def distance_statistics(values, scale, weights=None):
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones(values.size) if weights is None else np.asarray(weights)
    finite = np.isfinite(values)
    result = {"count": int(values.size), "nonfinite": int((~finite).sum())}
    if not values.size or not finite.all():
        return {**result, "status": "empty" if not values.size else "nonfinite",
                "absolute": None, "relative": None}
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    absolute = {
        "max": float(values.max()),
        "rms": float(np.sqrt(np.sum(weights * values**2) / weights.sum())),
        "p99": float(values[order[min(np.searchsorted(cumulative, .99), len(order)-1)]]),
    }
    return {**result, "status": "ok", "absolute": absolute,
            "relative": {k: v / scale for k, v in absolute.items()}}


def network_residual(restored, reference, original_count, scale):
    delta = np.asarray(restored) - np.asarray(reference)
    distances = np.linalg.norm(delta, axis=1)
    groups = {"original": slice(0, original_count),
              "auxiliary": slice(original_count, None), "all": slice(None)}
    return {
        "groups": {k: distance_statistics(distances[s], scale) for k, s in groups.items()},
        "legacy_max_abs_coordinate": float(np.max(np.abs(delta))),
        "weighting": "uniform vertices", "normalizer": "LUV0",
    }, delta


def sd_metrics(vertices, faces, uv):
    v = torch.as_tensor(vertices, dtype=torch.float64)
    f = torch.as_tensor(faces, dtype=torch.long)
    u = torch.as_tensor(uv, dtype=torch.float64)
    with torch.no_grad():
        regularized = symmetric_dirichlet_per_face(v, f, u).numpy()
        legacy = compute_distortion_metrics(vertices, faces, uv)
    tri = vertices[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    areas = .5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    # QR gives a local orthonormal 3D tangent frame without an absolute height floor.
    _, source = np.linalg.qr(np.stack((e1, e2), axis=-1), mode="reduced")
    target = np.stack((uv[faces[:, 1]]-uv[faces[:, 0]],
                       uv[faces[:, 2]]-uv[faces[:, 0]]), axis=-1)
    jacobian = np.swapaxes(np.linalg.solve(np.swapaxes(source, -1, -2),
                                         np.swapaxes(target, -1, -2)), -1, -2)
    sv = np.linalg.svd(jacobian, compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        strict = np.sum(sv**2 + sv**-2, axis=1)
    def summary(x):
        finite = np.isfinite(x)
        return {"area_weighted_mean": float(np.sum(x * areas)/areas.sum()),
                "p95_face_count": float(np.quantile(x, .95)),
                "p99_face_count": float(np.quantile(x, .99)),
                "max": float(x.max()), "nonfinite_faces": int((~finite).sum())}
    legacy_summary = {"area_weighted_mean": legacy["symmetric_dirichlet_area_weighted_mean"],
                      "p95_face_count": legacy["symmetric_dirichlet_p95"],
                      "p99_face_count": legacy["symmetric_dirichlet_p99"],
                      "max": legacy["symmetric_dirichlet_max"]}
    return {"regularized_f64": summary(regularized), "strict_qr_f64": summary(strict),
            "legacy_regularized_f32": legacy_summary, "faces": "original_only"}, {
                "sd_regularized": regularized, "sd_strict": strict, "area3d": areas,
                "singular_values": sv}


def orientation_signs(uv, faces):
    """Filtered signs; uncertain determinants use exact rational input coordinates."""
    tri = uv[faces]
    ax = tri[:, 1, 0]-tri[:, 0, 0]
    ay = tri[:, 1, 1]-tri[:, 0, 1]
    bx = tri[:, 2, 0]-tri[:, 0, 0]
    by = tri[:, 2, 1]-tri[:, 0, 1]
    determinant = ax*by-ay*bx
    finite = np.isfinite(tri).all(axis=(1, 2)) & np.isfinite(determinant)
    signs = np.zeros(len(faces), dtype=np.int8)
    signs[finite] = np.sign(determinant[finite]).astype(np.int8)
    uncertain = finite & (np.abs(determinant) <= 8*np.finfo(float).eps*(np.abs(ax*by)+np.abs(ay*bx)))
    for i in np.flatnonzero(uncertain):
        p, q, r = [[Fraction(float(x)) for x in row] for row in tri[i]]
        exact = (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        signs[i] = (exact > 0)-(exact < 0)
    return signs, finite, determinant/2


def validity(uv, faces, *, intersections=True):
    signs, finite, areas = orientation_signs(uv, faces)
    result = {"flipped": int(((signs < 0) & finite).sum()),
              "degenerate": int(((signs == 0) & finite).sum()),
              "nonfinite_faces": int((~finite).sum()),
              "legacy_num_flipped": int(((areas <= 1e-12) | ~finite).sum()),
              "minimum_signed_area": float(areas.min()) if finite.all() else None,
              "intersection_pairs": None, "intersection_status": "not_requested"}
    if not intersections:
        return result, np.empty((0, 2), dtype=np.int64)
    if not finite.all() or np.any(signs == 0):
        result["intersection_status"] = "undefined_degenerate_or_nonfinite"
        return result, np.empty((0, 2), dtype=np.int64)
    polygons = shapely.polygons(uv[faces])
    tree = STRtree(polygons)
    hits = []
    # A fixed query chunk bounds candidates by 32*F, not F squared.
    for start in range(0, len(faces), 32):
        pair = tree.query(polygons[start:start+32], predicate="intersects")
        i, j = pair[0]+start, pair[1]
        keep = j > i
        i, j = i[keep], j[keep]
        equal = faces[i, :, None] == faces[j, None, :]
        shared_i, shared_j = equal.any(axis=2), equal.any(axis=1)
        shared_count = shared_i.sum(axis=1)
        illegal = (shared_count == 0) | (shared_count == 3)
        edge = shared_count == 2
        if edge.any():
            common = faces[i[edge]][shared_i[edge]].reshape(-1, 2)
            opposite_i = faces[i[edge]][~shared_i[edge]]
            opposite_j = faces[j[edge]][~shared_j[edge]]
            si, _, _ = orientation_signs(uv, np.column_stack((common, opposite_i)))
            sj, _, _ = orientation_signs(uv, np.column_stack((common, opposite_j)))
            illegal[edge] = si == sj
        vertex = shared_count == 1
        if vertex.any():
            common = faces[i[vertex]][shared_i[vertex]]
            overlap = shapely.intersection(polygons[i[vertex]], polygons[j[vertex]])
            illegal[vertex] = ~shapely.is_empty(shapely.difference(overlap, shapely.points(uv[common])))
        hits.extend(zip(i[illegal].tolist(), j[illegal].tolist()))
    result.update(intersection_pairs=len(hits), intersection_status="complete")
    return result, np.asarray(hits, dtype=np.int64).reshape(-1, 2)


INTERIOR_BARY = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
EDGE_BARY = np.array([[1e-8, .5, .5-1e-8], [.5-1e-8, 1e-8, .5], [.5, .5-1e-8, 1e-8]])


def geometric_roundtrip(vertices, faces, native_uv, scale, *, decode_uv=None,
                        decode_vertices=None, near_edges=False, quantize_queries=False):
    decode_uv = native_uv if decode_uv is None else decode_uv
    decode_vertices = vertices if decode_vertices is None else decode_vertices
    bary = EDGE_BARY if near_edges else INTERIOR_BARY
    face_ids = np.repeat(np.arange(len(faces)), len(bary))
    weights = np.tile(bary, (len(faces), 1))
    tri = vertices[faces[face_ids]]
    points = np.einsum("ni,nij->nj", weights, tri)
    edges = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1)
    q, r = np.linalg.qr(edges, mode="reduced")
    recovered = np.linalg.solve(r, np.einsum("nji,nj->ni", q, points-tri[:, 0])[..., None])[..., 0]
    source_bary = np.column_stack((1-recovered.sum(axis=1), recovered))
    queries = np.einsum("ni,nij->nj", source_bary, native_uv[faces[face_ids]])
    if quantize_queries:
        origin, extent = native_uv.min(axis=0), np.ptp(native_uv, axis=0).max()
        queries = ((queries-origin)/extent).astype(np.float32).astype(float)*extent+origin
    def reconstruct(ids, query):
        uv_tri = decode_uv[faces[ids]]
        du = np.stack((uv_tri[:, 1]-uv_tri[:, 0], uv_tri[:, 2]-uv_tri[:, 0]), axis=-1)
        coordinates = np.linalg.solve(du, (query-uv_tri[:, 0])[..., None])[..., 0]
        b = np.column_stack((1-coordinates.sum(axis=1), coordinates))
        return np.einsum("ni,nij->nj", b, decode_vertices[faces[ids]])
    known = reconstruct(face_ids, queries)
    area = .5*np.linalg.norm(np.cross(edges[:, :, 0], edges[:, :, 1]), axis=1)
    polygons = shapely.polygons(decode_uv[faces])
    tree = STRtree(polygons)
    located = np.full(len(points), -1, dtype=np.int64)
    candidate_counts = np.zeros(len(points), dtype=np.int32)
    for start in range(0, len(points), 128):
        pair = tree.query(shapely.points(queries[start:start+128]), predicate="covered_by")
        if pair.size:
            rows = pair[0]+start
            np.add.at(candidate_counts, rows, 1)
            # Multiple valid shared-edge hits are retained as an ambiguity count.
            for row, face in zip(rows, pair[1]):
                if located[row] == -1 or face < located[row]:
                    located[row] = face
    found = located >= 0
    located_error = np.full(len(points), np.nan)
    located_error[found] = np.linalg.norm(reconstruct(located[found], queries[found])-points[found], axis=1)
    known_error = np.linalg.norm(known-points, axis=1)
    return {
        "sampling": "three_near_edge_points_per_face" if near_edges else "three_interior_points_per_face",
        "normalizer": "D3D", "weighting": "original_3d_face_area_equal_samples",
        "known_face": distance_statistics(known_error, scale, area),
        "located": distance_statistics(located_error, scale, area),
        "located_success_only": distance_statistics(located_error[found], scale, area[found]),
        "missing": int((~found).sum()), "multiple_candidates": int((candidate_counts > 1).sum()),
        "different_face": int(((located != face_ids) & found).sum()),
        "sampled_not_continuous_bound": True,
    }, {"face_ids": face_ids, "input_barycentric": weights,
         "known_error": known_error, "located_error": located_error,
         "located_face_ids": located, "candidate_counts": candidate_counts}
