from __future__ import annotations

import torch

from research.mesh_pl_nvp.batched_coupling import BatchedMeshCouplingFlow
from research.mesh_pl_nvp.mesh_coupling import (
    boundary_vertices,
    diagnose_embedding,
    make_grid_triangulation,
)
from research.mesh_pl_nvp.scaffold import build_outer_scaffold


def test_scaffold_turns_original_boundary_into_interior_vertices() -> None:
    uv, faces = make_grid_triangulation(6, 6)
    vertices = torch.column_stack((uv, 0.1 * uv[:, 0] * uv[:, 1]))
    source_boundary = boundary_vertices(faces, uv.shape[0])
    scaffold = build_outer_scaffold(vertices, faces, uv, scale=1.4)
    extended_boundary = boundary_vertices(scaffold.faces, scaffold.uv.shape[0])

    assert torch.all(~extended_boundary[: uv.shape[0]])
    assert torch.all(extended_boundary[scaffold.outer_boundary])
    assert torch.all(source_boundary[scaffold.original_boundary])
    diagnostics = diagnose_embedding(scaffold.uv, scaffold.faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0


def test_scaffold_flow_moves_source_boundary_and_remains_invertible() -> None:
    torch.manual_seed(8)
    uv, faces = make_grid_triangulation(6, 6)
    vertices = torch.column_stack((uv, 0.1 * uv[:, 0] * uv[:, 1]))
    scaffold = build_outer_scaffold(vertices, faces, uv, scale=1.4)
    model = BatchedMeshCouplingFlow(
        scaffold.vertices_3d, scaffold.faces, scaffold.uv, cycles=2, hidden_dim=12
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.015 * torch.randn_like(parameter))
    mapped = model()
    movement = torch.linalg.vector_norm(
        mapped[scaffold.original_boundary] - scaffold.uv[scaffold.original_boundary], dim=-1
    )
    assert float(movement.max().detach()) > 1.0e-6
    diagnostics = diagnose_embedding(mapped, scaffold.faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0
    restored = model(mapped, inverse=True)
    assert float(torch.max(torch.abs(restored - scaffold.uv)).detach()) < 2.0e-6
