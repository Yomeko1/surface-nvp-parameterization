from __future__ import annotations

import torch

from research.mesh_pl_nvp.batched_coupling import (
    BatchedMeshCouplingFlow,
    batched_polygon_vertex_mean,
)
from research.mesh_pl_nvp.radial_polytope import (
    halfplanes_from_ccw_polygon,
    polygon_vertex_mean,
)
from research.mesh_pl_nvp.mesh_coupling import diagnose_embedding, make_grid_triangulation


def _lift(uv: torch.Tensor) -> torch.Tensor:
    return torch.column_stack((uv, 0.12 * torch.sin(3.0 * uv[:, 0] + uv[:, 1])))


def test_batched_polygon_center_matches_reference_for_mixed_valence() -> None:
    polygons = [
        torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64),
        torch.tensor(
            [[-1.4, -0.5], [0.2, -1.1], [1.3, -0.2], [0.8, 1.1], [-0.7, 1.3]],
            dtype=torch.float64,
        ),
    ]
    halfplanes = [halfplanes_from_ccw_polygon(polygon) for polygon in polygons]
    width = max(A.shape[0] for A, _ in halfplanes)
    A_batch = torch.zeros((len(polygons), width, 2), dtype=torch.float64)
    b_batch = torch.zeros((len(polygons), width), dtype=torch.float64)
    mask = torch.zeros((len(polygons), width), dtype=torch.bool)
    expected = []
    for row, (A, b) in enumerate(halfplanes):
        A_batch[row, : A.shape[0]] = A
        b_batch[row, : b.shape[0]] = b
        mask[row, : A.shape[0]] = True
        expected.append(polygon_vertex_mean(A, b))
    actual = batched_polygon_vertex_mean(A_batch, b_batch, mask)
    assert torch.allclose(actual, torch.stack(expected), atol=1.0e-12, rtol=0.0)


def test_batched_identity_has_nonzero_training_gradient() -> None:
    uv, faces = make_grid_triangulation(6, 6)
    model = BatchedMeshCouplingFlow(_lift(uv), faces, uv, cycles=2, hidden_dim=12).double()
    mapped, diagnostics = model(return_diagnostics=True)
    assert torch.allclose(mapped, uv, atol=2.0e-11, rtol=0.0)
    assert float(diagnostics.q_values.max().detach()) < 1.0
    assert diagnostics.vertex_ids.shape == diagnostics.q_values.shape
    assert diagnostics.layer_ids.shape == diagnostics.q_values.shape
    assert torch.equal(torch.unique(diagnostics.vertex_ids), torch.nonzero(~model.boundary).flatten())
    assert int(diagnostics.layer_ids.max()) < model.cycles * model.color_count
    loss = torch.sum(mapped.square() * torch.tensor([1.0, 1.4], dtype=uv.dtype))
    loss.backward()
    final_layer_gradient = sum(
        float(conditioner.network[-1].weight.grad.square().sum())
        for conditioner in model.conditioners
    )
    assert final_layer_gradient > 0.0


def test_batched_flow_is_legal_and_invertible_after_parameter_perturbation() -> None:
    torch.manual_seed(17)
    uv, faces = make_grid_triangulation(6, 6)
    model = BatchedMeshCouplingFlow(_lift(uv), faces, uv, cycles=2, hidden_dim=12).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.025 * torch.randn_like(parameter))
    mapped = model()
    diagnostics = diagnose_embedding(mapped, faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0
    restored = model(mapped, inverse=True)
    assert float(torch.max(torch.abs(restored - uv)).detach()) < 2.0e-6


def test_local_geometry_conditioner_preserves_identity_and_inverse() -> None:
    uv, faces = make_grid_triangulation(6, 6)
    model = BatchedMeshCouplingFlow(
        _lift(uv), faces, uv, cycles=2, hidden_dim=12, feature_set="local-geometry"
    ).double()
    assert model.local_geometry_features.shape == (uv.shape[0], 6)
    mapped = model()
    assert torch.allclose(mapped, uv, atol=2.0e-11, rtol=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    mapped = model()
    restored = model(mapped, inverse=True)
    assert float(torch.max(torch.abs(restored - uv)).detach()) < 2.0e-6
