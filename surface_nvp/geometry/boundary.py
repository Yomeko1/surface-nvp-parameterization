from __future__ import annotations

from collections import defaultdict

import numpy as np

from .topology import build_edge_faces


def boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edge_faces = build_edge_faces(faces)
    return [edge for edge, fids in edge_faces.items() if len(fids) == 1]


def extract_boundary_loop(faces: np.ndarray) -> list[int]:
    edges = boundary_edges(faces)
    if not edges:
        raise ValueError("mesh has no boundary; this pipeline expects disk topology")
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(v) != 2 for v in adj.values()):
        raise ValueError("boundary is not a single manifold loop")
    start = min(adj)
    loop = [start]
    prev = None
    curr = start
    while True:
        candidates = adj[curr]
        nxt = candidates[0] if candidates[0] != prev else candidates[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, curr = curr, nxt
        if len(loop) > len(adj):
            raise ValueError("failed to close boundary loop")
    if len(loop) != len(adj):
        raise ValueError("multiple boundary loops are not supported")
    return loop


def boundary_mask(num_vertices: int, loop: list[int]) -> np.ndarray:
    mask = np.zeros(num_vertices, dtype=bool)
    mask[np.asarray(loop, dtype=np.int64)] = True
    return mask
