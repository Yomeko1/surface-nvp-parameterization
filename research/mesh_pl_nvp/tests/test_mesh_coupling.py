from __future__ import annotations

import torch

from research.mesh_pl_nvp.mesh_coupling import (
    boundary_vertices,
    coupling_cycle,
    diagnose_embedding,
    greedy_vertex_coloring,
    make_grid_triangulation,
    vertex_adjacency,
)


def test_coloring_produces_independent_interior_sets() -> None:
    vertices, faces = make_grid_triangulation(7, 6)
    colors = greedy_vertex_coloring(faces, vertices.shape[0])
    boundary = boundary_vertices(faces, vertices.shape[0])
    adjacency = vertex_adjacency(faces, vertices.shape[0])
    for color in range(int(colors.max()) + 1):
        active = torch.nonzero((colors == color) & ~boundary).flatten().tolist()
        active_set = set(active)
        assert all(not adjacency[i].intersection(active_set) for i in active)


def test_full_coupling_cycle_is_legal_and_explicitly_invertible() -> None:
    initial, faces = make_grid_triangulation(7, 7)
    deformed = coupling_cycle(initial, faces)

    diagnostics = diagnose_embedding(deformed, faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0
    assert diagnostics.minimum_double_area > 0.0

    restored = coupling_cycle(deformed, faces, inverse=True)
    assert float(torch.max(torch.abs(restored - initial))) < 1.0e-8


def test_repeated_aggressive_cycles_remain_legal_with_bounded_inverse_error() -> None:
    initial, faces = make_grid_triangulation(7, 7)
    deformed = initial
    for _ in range(3):
        deformed = coupling_cycle(deformed, faces)

    diagnostics = diagnose_embedding(deformed, faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0
    assert diagnostics.minimum_double_area > 1.0e-6

    restored = deformed
    for _ in range(3):
        restored = coupling_cycle(restored, faces, inverse=True)
    # This intentionally severe test reduces the smallest face area by roughly
    # three orders of magnitude, so it also measures inverse conditioning.
    assert float(torch.max(torch.abs(restored - initial))) < 5.0e-6
