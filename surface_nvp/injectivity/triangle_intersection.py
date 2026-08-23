from __future__ import annotations

import numpy as np
import torch


def _orient(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a, b, p, eps=1e-12) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orient(a, b, p)) <= eps
    )


def segments_intersect(a, b, c, d, eps=1e-12) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return _on_segment(a, b, c, eps) or _on_segment(a, b, d, eps) or _on_segment(c, d, a, eps) or _on_segment(c, d, b, eps)


def point_in_triangle(p, tri, eps=1e-12) -> bool:
    a, b, c = tri
    o1 = _orient(a, b, p)
    o2 = _orient(b, c, p)
    o3 = _orient(c, a, p)
    return (o1 >= -eps and o2 >= -eps and o3 >= -eps) or (o1 <= eps and o2 <= eps and o3 <= eps)


def triangles_intersect(t0: np.ndarray, t1: np.ndarray) -> bool:
    for i in range(3):
        a, b = t0[i], t0[(i + 1) % 3]
        for j in range(3):
            c, d = t1[j], t1[(j + 1) % 3]
            if segments_intersect(a, b, c, d):
                return True
    return point_in_triangle(t0[0], t1) or point_in_triangle(t1[0], t0)


def find_triangle_intersections(
    uv: np.ndarray,
    faces: np.ndarray,
    ignore_adjacent: bool = True,
    tile_size: int = 512,
) -> list[tuple[int, int]]:
    """Return exact UV triangle intersections, memory-bounded via block processing.

    Preserves exact global detection (every non-adjacent, bbox-overlapping pair is
    tested exactly), but enumerates upper-triangular pairs in bounded tiles so it
    neither materializes an O(F^2) pair array nor loops over all pairs in Python.
    """
    tris = uv[faces]
    boxes_min = tris.min(axis=1)
    boxes_max = tris.max(axis=1)
    n = int(faces.shape[0])
    hits: list[tuple[int, int]] = []
    if n < 2:
        return hits

    for i0 in range(0, n, tile_size):
        i1 = min(i0 + tile_size, n)
        for j0 in range(i0, n, tile_size):
            j1 = min(j0 + tile_size, n)
            if i0 == j0:
                diag = i1 - i0
                if diag < 2:
                    continue
                ib, jb = np.triu_indices(diag, k=1)
                pi = (ib + i0).astype(np.int64)
                pj = (jb + j0).astype(np.int64)
            else:
                ni = i1 - i0
                nj = j1 - j0
                gi, gj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
                pi = (gi.ravel() + i0).astype(np.int64)
                pj = (gj.ravel() + j0).astype(np.int64)

            if ignore_adjacent:
                fi = faces[pi]
                fj = faces[pj]
                adjacent = (fi[:, :, None] == fj[:, None, :]).any(axis=(1, 2))
                if adjacent.all():
                    continue
                pi = pi[~adjacent]
                pj = pj[~adjacent]

            bmin0 = boxes_min[pi]
            bmax0 = boxes_max[pi]
            bmin1 = boxes_min[pj]
            bmax1 = boxes_max[pj]
            ov = ~((bmax0 < bmin1).any(axis=1) | (bmax1 < bmin0).any(axis=1))
            if not ov.any():
                continue
            pi = pi[ov]
            pj = pj[ov]

            for a, b in zip(pi.tolist(), pj.tolist()):
                if triangles_intersect(tris[a], tris[b]):
                    hits.append((a, b))
    return hits


def count_triangle_intersections_torch(
    uv: torch.Tensor,
    faces: torch.Tensor,
    ignore_adjacent: bool = True,
    batch_size: int = 262144,
    tile_size: int = 1024,
    eps: float = 1e-12,
) -> int:
    """Count UV triangle intersections with torch, memory-bounded block processing.

    This is intended for training-time validation. It keeps the exact segment
    intersection tests on torch tensors, but enumerates triangle pairs in
    upper-triangular tiles so it never materializes an O(F^2) pair tensor. This
    avoids CUDA out-of-memory on large meshes while preserving exact global
    detection: every non-adjacent, axis-aligned-bounding-box-overlapping pair is
    still tested exactly.
    """
    tris = uv[faces]
    num_faces = int(faces.shape[0])
    if num_faces < 2:
        return 0

    total = 0
    dev = uv.device
    for i0 in range(0, num_faces, tile_size):
        i1 = min(i0 + tile_size, num_faces)
        for j0 in range(i0, num_faces, tile_size):
            j1 = min(j0 + tile_size, num_faces)
            if i0 == j0:
                diag = i1 - i0
                if diag < 2:
                    continue
                pair_i, pair_j = torch.triu_indices(diag, diag, offset=1, device=dev)
                pair_i = pair_i + i0
                pair_j = pair_j + j0
            else:
                ni = i1 - i0
                nj = j1 - j0
                gi, gj = torch.meshgrid(
                    torch.arange(ni, device=dev), torch.arange(nj, device=dev), indexing="ij"
                )
                pair_i = (gi.reshape(-1) + i0).to(torch.long)
                pair_j = (gj.reshape(-1) + j0).to(torch.long)

            if pair_i.numel() == 0:
                continue

            if ignore_adjacent:
                fi = faces[pair_i]
                fj = faces[pair_j]
                adjacent = (fi[:, :, None] == fj[:, None, :]).any(dim=(1, 2))
                pair_i = pair_i[~adjacent]
                pair_j = pair_j[~adjacent]
                if pair_i.numel() == 0:
                    continue

            for start in range(0, pair_i.numel(), batch_size):
                end = min(start + batch_size, pair_i.numel())
                t0 = tris[pair_i[start:end]]
                t1 = tris[pair_j[start:end]]
                min0, max0 = t0.amin(dim=1), t0.amax(dim=1)
                min1, max1 = t1.amin(dim=1), t1.amax(dim=1)
                overlap = ~((max0 < min1).any(dim=1) | (max1 < min0).any(dim=1))
                if not overlap.any():
                    continue
                t0 = t0[overlap]
                t1 = t1[overlap]
                total += int(_triangles_intersect_torch(t0, t1, eps=eps).sum().item())
    return total


def _orient_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _on_segment_torch(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, eps: float) -> torch.Tensor:
    return (
        (torch.minimum(a[..., 0], b[..., 0]) - eps <= p[..., 0])
        & (p[..., 0] <= torch.maximum(a[..., 0], b[..., 0]) + eps)
        & (torch.minimum(a[..., 1], b[..., 1]) - eps <= p[..., 1])
        & (p[..., 1] <= torch.maximum(a[..., 1], b[..., 1]) + eps)
        & (_orient_torch(a, b, p).abs() <= eps)
    )


def _segments_intersect_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, eps: float) -> torch.Tensor:
    o1 = _orient_torch(a, b, c)
    o2 = _orient_torch(a, b, d)
    o3 = _orient_torch(c, d, a)
    o4 = _orient_torch(c, d, b)
    proper = (o1 * o2 < -eps) & (o3 * o4 < -eps)
    touch = _on_segment_torch(a, b, c, eps) | _on_segment_torch(a, b, d, eps) | _on_segment_torch(c, d, a, eps) | _on_segment_torch(c, d, b, eps)
    return proper | touch


def _point_in_triangle_torch(p: torch.Tensor, tri: torch.Tensor, eps: float) -> torch.Tensor:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    o1 = _orient_torch(a, b, p)
    o2 = _orient_torch(b, c, p)
    o3 = _orient_torch(c, a, p)
    return ((o1 >= -eps) & (o2 >= -eps) & (o3 >= -eps)) | ((o1 <= eps) & (o2 <= eps) & (o3 <= eps))


def _triangles_intersect_torch(t0: torch.Tensor, t1: torch.Tensor, eps: float) -> torch.Tensor:
    hit = torch.zeros(t0.shape[0], dtype=torch.bool, device=t0.device)
    for i in range(3):
        a, b = t0[:, i], t0[:, (i + 1) % 3]
        for j in range(3):
            c, d = t1[:, j], t1[:, (j + 1) % 3]
            hit |= _segments_intersect_torch(a, b, c, d, eps)
    hit |= _point_in_triangle_torch(t0[:, 0], t1, eps)
    hit |= _point_in_triangle_torch(t1[:, 0], t0, eps)
    return hit
