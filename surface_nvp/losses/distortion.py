from __future__ import annotations

import torch


def _local_2d(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces]
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1 = p1 - p0
    e2 = p2 - p0
    l1 = torch.linalg.norm(e1, dim=-1).clamp_min(1e-12)
    x2 = (e1 * e2).sum(dim=-1) / l1
    y2_sq = torch.linalg.norm(e2, dim=-1).pow(2) - x2.pow(2)
    y2 = torch.sqrt(torch.clamp(y2_sq, min=1e-12))
    out = torch.zeros((faces.shape[0], 3, 2), dtype=vertices.dtype, device=vertices.device)
    out[:, 1, 0] = l1
    out[:, 2, 0] = x2
    out[:, 2, 1] = y2
    return out


def symmetric_dirichlet_per_face(vertices: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    x = _local_2d(vertices, faces)
    u = uv[faces]
    dx = torch.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], dim=-1)
    du = torch.stack([u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]], dim=-1)
    j = du @ torch.linalg.inv(dx)
    singular_values = torch.linalg.svdvals(j)
    inverse_singular_values = (singular_values + 1e-4).reciprocal()
    return (singular_values.pow(2) + inverse_singular_values.pow(2)).sum(dim=-1)


def symmetric_dirichlet_loss(vertices: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    per_face = symmetric_dirichlet_per_face(vertices, faces, uv)
    tri = vertices[faces]
    areas = 0.5 * torch.linalg.norm(torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), dim=-1)
    return (per_face * areas).sum() / areas.sum().clamp_min(1e-12)


def jacobian_determinants(vertices: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    x = _local_2d(vertices, faces)
    u = uv[faces]
    dx = torch.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], dim=-1)
    du = torch.stack([u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]], dim=-1)
    return torch.det(du @ torch.linalg.inv(dx))


def jacobian_barrier_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uv: torch.Tensor,
    reference_scale: torch.Tensor,
    margin: float = 1e-2,
    temperature: float = 1e-2,
) -> torch.Tensor:
    normalized_det = jacobian_determinants(vertices, faces, uv) / reference_scale.clamp_min(1e-12)
    violation = torch.nn.functional.softplus((margin - normalized_det) / temperature) * temperature
    return violation.pow(2).mean()
