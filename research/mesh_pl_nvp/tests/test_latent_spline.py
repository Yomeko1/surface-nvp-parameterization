from __future__ import annotations

import torch

from research.mesh_pl_nvp.batched_coupling import BatchedMeshCouplingFlow
from research.mesh_pl_nvp.latent_spline import (
    LatentSplineConditioner,
    batched_legal_axis_intervals,
    interval_spline_coupling,
)
from research.mesh_pl_nvp.mesh_coupling import diagnose_embedding, make_grid_triangulation
from research.mesh_pl_nvp.scaffold import build_outer_scaffold


def test_latent_spline_starts_at_identity_and_has_an_exact_inverse() -> None:
    torch.manual_seed(71)
    latent = torch.randn(17, dtype=torch.float64)
    lower = -2.0 - 0.1 * torch.rand(17, dtype=torch.float64)
    upper = 2.0 + 0.1 * torch.rand(17, dtype=torch.float64)
    values = lower + torch.sigmoid(latent) * (upper - lower)
    retained = torch.randn(17, dtype=torch.float64)
    features = torch.randn(17, 8, dtype=torch.float64)
    conditioner = LatentSplineConditioner(
        feature_dim=8,
        hidden_dim=16,
        num_bins=8,
        tail_bound=8.0,
        dtype=torch.float64,
    )
    identity = interval_spline_coupling(
        values, lower, upper, retained, features, conditioner
    )
    assert torch.allclose(identity, values, atol=2.0e-12, rtol=2.0e-12)

    with torch.no_grad():
        for parameter in conditioner.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    mapped = interval_spline_coupling(
        values, lower, upper, retained, features, conditioner
    )
    mapped.square().sum().backward()
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in conditioner.parameters()
    )
    restored = interval_spline_coupling(
        mapped.detach(),
        lower,
        upper,
        retained,
        features,
        conditioner,
        inverse=True,
    )
    assert float(torch.max(torch.abs(restored - values)).detach()) < 2.0e-10


def test_axis_intervals_match_a_rectangle_cross_section() -> None:
    points = torch.tensor([[0.25, -0.4]], dtype=torch.float64)
    A = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]],
        dtype=torch.float64,
    )
    b = torch.tensor([[2.0, 1.0, 3.0, 4.0]], dtype=torch.float64)
    mask = torch.ones((1, 4), dtype=torch.bool)
    lower, upper = batched_legal_axis_intervals(
        points, A, b, mask, transformed_index=0
    )
    assert torch.equal(lower, torch.tensor([-1.0], dtype=torch.float64))
    assert torch.equal(upper, torch.tensor([2.0], dtype=torch.float64))


def test_batched_spline_mesh_flow_preserves_embedding_and_round_trip() -> None:
    torch.manual_seed(73)
    uv, faces = make_grid_triangulation(7, 7)
    xyz = torch.column_stack((uv, 0.07 * torch.sin(uv[:, 0] + uv[:, 1])))
    scaffold = build_outer_scaffold(xyz, faces, uv, scale=1.25)
    model = BatchedMeshCouplingFlow(
        scaffold.vertices_3d,
        scaffold.faces,
        scaffold.uv,
        cycles=1,
        hidden_dim=16,
        radial_map="atanh",
        latent_transform="spline",
        spline_bins=8,
        spline_bound=8.0,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.002 * torch.randn_like(parameter))
    mapped = model()
    diagnostics = diagnose_embedding(mapped, scaffold.faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0
    mapped.square().sum().backward()
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
    restored = model(mapped.detach(), inverse=True)
    assert float(torch.max(torch.abs(restored - scaffold.uv)).detach()) < 2.0e-8
