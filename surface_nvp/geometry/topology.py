from __future__ import annotations

from collections import defaultdict

import numpy as np


def build_edge_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_faces[key].append(fi)
    return dict(edge_faces)


def build_vertex_neighbors(faces: np.ndarray, num_vertices: int) -> list[list[int]]:
    neighbors = [set() for _ in range(num_vertices)]
    for face in faces:
        a, b, c = map(int, face)
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return [sorted(n) for n in neighbors]


def adjacent_face_pairs(faces: np.ndarray) -> set[tuple[int, int]]:
    edge_faces = build_edge_faces(faces)
    pairs = set()
    for fids in edge_faces.values():
        if len(fids) == 2:
            a, b = sorted(fids)
            pairs.add((a, b))
    return pairs
