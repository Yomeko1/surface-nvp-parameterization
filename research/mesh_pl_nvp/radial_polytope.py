"""Explicit radial homeomorphism between R^2 and a bounded convex polygon.

The polygon is represented as ``K = {x | A x < b}``.  ``to_polytope`` maps
R^2 to the interior of K, while ``from_polytope`` is its analytical inverse.
Both functions are differentiable almost everywhere; the exceptional set is
where a ray from the center hits two polygon edges simultaneously.

This module is deliberately independent from ``surface_nvp`` and the v2.4
training pipeline.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def _require_shapes(A: Tensor, b: Tensor, center: Tensor) -> None:
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError(f"A must have shape (m, 2), got {tuple(A.shape)}")
    if b.shape != (A.shape[0],):
        raise ValueError(f"b must have shape ({A.shape[0]},), got {tuple(b.shape)}")
    if center.shape != (2,):
        raise ValueError(f"center must have shape (2,), got {tuple(center.shape)}")
    if A.device != b.device or A.device != center.device:
        raise ValueError("A, b, and center must be on the same device")
    if A.dtype != b.dtype or A.dtype != center.dtype:
        raise ValueError("A, b, and center must have the same dtype")


def halfplanes_from_ccw_polygon(vertices: Tensor) -> tuple[Tensor, Tensor]:
    """Convert a strictly convex CCW polygon to normalized ``A x <= b`` rows."""
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        raise ValueError("vertices must have shape (n, 2), n >= 3")

    nxt = torch.roll(vertices, shifts=-1, dims=0)
    edges = nxt - vertices
    signed_area2 = torch.sum(
        vertices[:, 0] * nxt[:, 1] - vertices[:, 1] * nxt[:, 0]
    )
    if float(signed_area2.detach()) <= 0.0:
        raise ValueError("polygon vertices must be counter-clockwise")

    # cross(edge, x - vertex) >= 0  <=>  [edge_y, -edge_x] x <= b.
    A = torch.stack((edges[:, 1], -edges[:, 0]), dim=-1)
    lengths = torch.linalg.vector_norm(A, dim=-1)
    if bool(torch.any(lengths <= torch.finfo(vertices.dtype).eps)):
        raise ValueError("polygon contains a zero-length edge")
    A = A / lengths[:, None]
    b = torch.sum(A * vertices, dim=-1)
    return A, b


def halfplane_polygon_vertices(
    A: Tensor,
    b: Tensor,
    *,
    parallel_tolerance: float = 1.0e-12,
    feasibility_tolerance: float = 1.0e-9,
) -> Tensor:
    """Return feasible pairwise line intersections of a bounded 2D polytope.

    Selection of the active constraint pairs is discrete, but the returned
    intersections retain gradients with respect to the selected rows of A,b.
    For a non-degenerate convex polygon these points are its vertices. Redundant
    constraints may yield duplicate vertices, which is harmless for computing
    a strict convex-combination center.
    """
    if A.ndim != 2 or A.shape[1] != 2 or b.shape != (A.shape[0],):
        raise ValueError("expected A=(m,2), b=(m,)")

    candidates: list[Tensor] = []
    for i, j in combinations(range(A.shape[0]), 2):
        ai, aj = A[i], A[j]
        det = ai[0] * aj[1] - ai[1] * aj[0]
        if abs(float(det.detach())) <= parallel_tolerance:
            continue
        point = torch.stack(
            (
                (b[i] * aj[1] - ai[1] * b[j]) / det,
                (ai[0] * b[j] - b[i] * aj[0]) / det,
            )
        )
        violation = A @ point - b
        if bool(torch.all(violation.detach() <= feasibility_tolerance)):
            candidates.append(point)

    if len(candidates) < 3:
        raise ValueError("half-planes do not define a bounded non-degenerate polygon")
    return torch.stack(candidates)


def polygon_vertex_mean(A: Tensor, b: Tensor) -> Tensor:
    """Choose an interior anchor using the mean of the polygon vertices."""
    vertices = halfplane_polygon_vertices(A, b)
    center = vertices.mean(dim=0)
    slack = b - A @ center
    scale = torch.maximum(torch.ones_like(slack), torch.abs(b))
    tolerance = 64.0 * torch.finfo(A.dtype).eps * scale
    if bool(torch.any(slack.detach() <= tolerance.detach())):
        raise ValueError("computed polygon center is not strictly interior")
    return center


def analytic_center(
    A: Tensor,
    b: Tensor,
    *,
    iterations: int = 24,
    backtracking_steps: int = 40,
) -> Tensor:
    """Compute the unique log-barrier center of a bounded polygon.

    The objective is ``-sum(log(b - A x))``. Newton iterations remain in the
    polygon interior and are composed of torch operations, so gradients flow
    through the accepted iterations. The vertex mean is used only as a feasible
    initializer; convergence to the unique analytic center removes most of its
    active-set sensitivity on symmetric or nearly degenerate one-rings.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    center = polygon_vertex_mean(A, b)
    for _ in range(iterations):
        slack = b - A @ center
        inverse_slack = slack.reciprocal()
        gradient = A.transpose(0, 1) @ inverse_slack
        hessian = A.transpose(0, 1) @ (A * inverse_slack.square()[:, None])
        newton_step = torch.linalg.solve(hessian, gradient)

        current_objective = -torch.log(slack).sum()
        directional_derivative = torch.dot(gradient, -newton_step)
        step_size = 1.0
        accepted = False
        for _ in range(backtracking_steps):
            candidate = center - step_size * newton_step
            candidate_slack = b - A @ candidate
            if bool(torch.all(candidate_slack.detach() > 0.0)):
                candidate_objective = -torch.log(candidate_slack).sum()
                armijo_bound = current_objective + 1.0e-4 * step_size * directional_derivative
                if float(candidate_objective.detach()) <= float(armijo_bound.detach()):
                    center = candidate
                    accepted = True
                    break
            step_size *= 0.5
        if not accepted:
            break
        if float(torch.linalg.vector_norm(newton_step).detach()) <= 1.0e-13:
            break
    return center


def ray_radius(
    directions: Tensor,
    A: Tensor,
    b: Tensor,
    center: Tensor,
    *,
    direction_tolerance: float | None = None,
) -> Tensor:
    """Distance from ``center`` to the polygon boundary along unit directions."""
    _require_shapes(A, b, center)
    if directions.shape[-1] != 2:
        raise ValueError("directions must have final dimension 2")
    if direction_tolerance is None:
        direction_tolerance = 32.0 * torch.finfo(A.dtype).eps

    slack = b - A @ center
    if bool(torch.any(slack.detach() <= 0.0)):
        raise ValueError("center must lie strictly inside the polygon")

    denominators = directions @ A.transpose(0, 1)
    candidates = slack / torch.where(
        denominators > direction_tolerance,
        denominators,
        torch.ones_like(denominators),
    )
    candidates = torch.where(
        denominators > direction_tolerance,
        candidates,
        torch.full_like(candidates, torch.inf),
    )
    radius = candidates.amin(dim=-1)
    if bool(torch.any(~torch.isfinite(radius.detach()))):
        raise ValueError("polygon is unbounded in at least one requested direction")
    return radius


def _radius_and_direction(delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    radius = torch.linalg.vector_norm(delta, dim=-1)
    tiny = 32.0 * torch.finfo(delta.dtype).eps
    nonzero = radius > tiny
    safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
    direction = delta / safe_radius[..., None]
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    direction = torch.where(nonzero[..., None], direction, fallback)
    return radius, direction, nonzero


def to_polytope(
    latent: Tensor,
    A: Tensor,
    b: Tensor,
    center: Tensor,
    *,
    radial_scale: float = 1.0,
) -> Tensor:
    """Map ``R^2`` bijectively into the interior of ``K={x | A x < b}``.

    ``radial_scale`` is the positive softsign scale. It changes conditioning,
    not the image or the bijectivity of the map.
    """
    _require_shapes(A, b, center)
    if latent.shape[-1] != 2:
        raise ValueError("latent must have final dimension 2")
    if radial_scale <= 0.0:
        raise ValueError("radial_scale must be positive")

    delta = latent - center
    radius, direction, nonzero = _radius_and_direction(delta)
    boundary_radius = ray_radius(direction, A, b, center)
    relative_radius = radius / boundary_radius
    squashed = radial_scale * relative_radius / (1.0 + radial_scale * relative_radius)
    mapped = center + (boundary_radius * squashed)[..., None] * direction
    # The radial formula has first-order limit c + radial_scale * (z-c) at
    # the center. Using that continuation avoids a zero-gradient identity
    # branch when a mesh vertex happens to equal the analytic center.
    center_limit = center + radial_scale * delta
    return torch.where(nonzero[..., None], mapped, center_limit)


def from_polytope(
    points: Tensor,
    A: Tensor,
    b: Tensor,
    center: Tensor,
    *,
    radial_scale: float = 1.0,
    membership_tolerance: float = 1.0e-10,
) -> Tensor:
    """Analytical inverse of :func:`to_polytope`."""
    _require_shapes(A, b, center)
    if points.shape[-1] != 2:
        raise ValueError("points must have final dimension 2")
    if radial_scale <= 0.0:
        raise ValueError("radial_scale must be positive")

    violation = points @ A.transpose(0, 1) - b
    if bool(torch.any(violation.detach() >= membership_tolerance)):
        worst = float(violation.detach().max())
        raise ValueError(f"points must be strictly inside the polygon; max violation={worst:.3e}")

    delta = points - center
    radius, direction, nonzero = _radius_and_direction(delta)
    boundary_radius = ray_radius(direction, A, b, center)
    fraction = radius / boundary_radius
    if bool(torch.any(fraction.detach() >= 1.0)):
        raise ValueError("point lies on or outside the polygon boundary")

    unsquashed = fraction / (radial_scale * (1.0 - fraction))
    latent = center + (boundary_radius * unsquashed)[..., None] * direction
    center_limit = center + delta / radial_scale
    return torch.where(nonzero[..., None], latent, center_limit)


def minimum_slack(points: Tensor, A: Tensor, b: Tensor) -> Tensor:
    """Return the minimum signed half-plane slack for each point."""
    return (b - points @ A.transpose(0, 1)).amin(dim=-1)
