from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import ConvexHull

from research.mesh_pl_nvp.radial_polytope import (
    analytic_center,
    from_polytope,
    halfplanes_from_ccw_polygon,
    minimum_slack,
    polygon_vertex_mean,
    to_polytope,
)


DTYPE = torch.float64


def _random_convex_polygon(seed: int, count: int = 30) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 2))
    points[:, 0] *= rng.uniform(0.25, 2.5)
    points[:, 1] *= rng.uniform(0.25, 2.5)
    hull = ConvexHull(points)
    return torch.tensor(points[hull.vertices], dtype=DTYPE)


def test_round_trip_and_strict_membership_on_random_polygons() -> None:
    generator = torch.Generator().manual_seed(7)
    for seed in range(8):
        polygon = _random_convex_polygon(seed)
        A, b = halfplanes_from_ccw_polygon(polygon)
        center = polygon_vertex_mean(A, b)
        latent = center + 4.0 * torch.randn((512, 2), generator=generator, dtype=DTYPE)
        points = to_polytope(latent, A, b, center)
        restored = from_polytope(points, A, b, center)
        assert float(minimum_slack(points, A, b).min()) > 0.0
        assert float(torch.max(torch.abs(restored - latent))) < 2.0e-10


def test_center_is_fixed_point() -> None:
    polygon = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [1.5, 1.0], [-1.0, 2.0]], dtype=DTYPE)
    A, b = halfplanes_from_ccw_polygon(polygon)
    center = polygon_vertex_mean(A, b)
    assert torch.equal(to_polytope(center, A, b, center), center)
    assert torch.equal(from_polytope(center, A, b, center), center)


def test_autograd_matches_finite_difference_away_from_rays_to_corners() -> None:
    polygon = torch.tensor([[-1.8, -1.0], [1.4, -1.2], [2.0, 0.7], [0.2, 1.8], [-1.4, 1.1]], dtype=DTYPE)
    A, b = halfplanes_from_ccw_polygon(polygon)
    center = polygon_vertex_mean(A, b)
    latent = (center + torch.tensor([0.37, -0.61], dtype=DTYPE)).requires_grad_(True)

    def mapping(value: torch.Tensor) -> torch.Tensor:
        return to_polytope(value, A, b, center)

    assert torch.autograd.gradcheck(mapping, (latent,), eps=1.0e-6, atol=2.0e-6, rtol=2.0e-5)
    jacobian = torch.autograd.functional.jacobian(mapping, latent)
    assert float(torch.linalg.det(jacobian)) > 0.0


def test_gradients_flow_through_polygon_constraints_and_center() -> None:
    polygon = torch.tensor(
        [[-1.0, -1.0], [1.3, -0.9], [1.6, 0.8], [0.1, 1.7], [-1.2, 0.9]],
        dtype=DTYPE,
        requires_grad=True,
    )
    A, b = halfplanes_from_ccw_polygon(polygon)
    center = polygon_vertex_mean(A, b)
    latent = center.detach() + torch.tensor([0.4, 0.2], dtype=DTYPE)
    loss = to_polytope(latent, A, b, center).square().sum()
    loss.backward()
    assert polygon.grad is not None
    assert bool(torch.all(torch.isfinite(polygon.grad)))
    assert float(torch.linalg.vector_norm(polygon.grad)) > 0.0


def test_analytic_center_is_interior_and_differentiable() -> None:
    polygon = torch.tensor(
        [[-2.0, -0.3], [1.8, -0.4], [2.1, 0.5], [0.6, 0.9], [-1.7, 0.7]],
        dtype=DTYPE,
        requires_grad=True,
    )
    A, b = halfplanes_from_ccw_polygon(polygon)
    center = analytic_center(A, b)
    assert float(minimum_slack(center, A, b).detach()) > 0.0
    center.square().sum().backward()
    assert polygon.grad is not None
    assert bool(torch.all(torch.isfinite(polygon.grad)))
