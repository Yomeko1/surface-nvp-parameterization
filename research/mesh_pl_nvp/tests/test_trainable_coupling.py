from __future__ import annotations

import torch

from research.mesh_pl_nvp.mesh_coupling import diagnose_embedding, make_grid_triangulation
from research.mesh_pl_nvp.trainable_coupling import MeshCouplingFlow


def _lift_to_3d(uv: torch.Tensor) -> torch.Tensor:
    z = 0.15 * torch.sin(2.0 * uv[:, 0]) * torch.cos(2.0 * uv[:, 1])
    return torch.column_stack((uv, z))


def test_zero_initialized_trainable_flow_is_identity_and_has_gradients() -> None:
    uv, faces = make_grid_triangulation(5, 5)
    vertices = _lift_to_3d(uv)
    model = MeshCouplingFlow(vertices, faces, uv, cycles=1, hidden_dim=12).double()
    mapped = model()
    assert torch.allclose(mapped, uv, atol=1.0e-12, rtol=0.0)

    loss = (mapped.square() * torch.tensor([1.0, 1.7], dtype=uv.dtype)).sum()
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient_norm > 0.0


def test_nontrivial_parameters_preserve_legality_and_invert() -> None:
    torch.manual_seed(4)
    uv, faces = make_grid_triangulation(5, 5)
    vertices = _lift_to_3d(uv)
    model = MeshCouplingFlow(vertices, faces, uv, cycles=1, hidden_dim=12).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.03 * torch.randn_like(parameter))
    mapped = model()
    diagnostics = diagnose_embedding(mapped, faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0
    restored = model(mapped, inverse=True)
    assert float(torch.max(torch.abs(restored - uv)).detach()) < 2.0e-7
