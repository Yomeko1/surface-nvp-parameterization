"""Opt-in multiring scaffold for boundary-mode experiments.

The v3.0 one-ring scaffold remains unchanged.  This research-only builder
inserts trainable transition rings before the fixed convex outer boundary.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .mesh_coupling import count_proper_edge_intersections, signed_double_areas
from .scaffold import ScaffoldMesh, ordered_boundary_loop


def _signed_polygon_area(points: Tensor) -> Tensor:
    following = torch.roll(points, shifts=-1, dims=0)
    return 0.5 * torch.sum(
        points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]
    )


def multiring_transition_profile(
    scaffold: ScaffoldMesh,
    *,
    rings: int,
    transition_exponent: float,
) -> tuple[Tensor, Tensor]:
    """Return intermediate ring indices and their source-motion weights.

    The scaffold geometry is ``(1-f) * source + f * outer`` with
    ``f=(ring_index/rings)**transition_exponent``.  Applying the matching
    displacement weight ``1-f`` makes a deformed ring remain on the same
    interpolation profile between the moving source and fixed outer ring.
    """
    if not isinstance(rings, int) or isinstance(rings, bool) or rings < 2:
        raise ValueError("multiring transition profile requires at least two rings")
    if transition_exponent <= 0.0 or not math.isfinite(transition_exponent):
        raise ValueError("transition exponent must be finite and positive")
    boundary_count = int(scaffold.original_boundary.numel())
    expected_vertex_count = scaffold.original_vertex_count + rings * boundary_count
    if scaffold.uv.shape[0] != expected_vertex_count:
        raise ValueError("scaffold vertex count does not match the requested ring count")
    all_rings = torch.arange(
        scaffold.original_vertex_count,
        expected_vertex_count,
        dtype=torch.long,
        device=scaffold.faces.device,
    ).reshape(rings, boundary_count)
    if not torch.equal(all_rings[-1], scaffold.outer_boundary):
        raise ValueError("the final inferred ring is not the fixed outer boundary")
    fractions = (
        torch.arange(
            1,
            rings,
            dtype=scaffold.uv.dtype,
            device=scaffold.uv.device,
        )
        / float(rings)
    ).pow(transition_exponent)
    return all_rings[:-1], 1.0 - fractions


def build_multiring_outer_scaffold(
    vertices_3d: Tensor,
    faces: Tensor,
    uv: Tensor,
    *,
    scale: float = 1.1,
    rings: int = 6,
    transition_exponent: float = 3.0,
) -> ScaffoldMesh:
    """Add trainable transition rings ending at one fixed convex ring."""
    if scale <= 1.0:
        raise ValueError("scaffold scale must be greater than one")
    if not isinstance(rings, int) or isinstance(rings, bool) or rings < 2:
        raise ValueError("multiring scaffold requires at least two rings")
    if transition_exponent <= 0.0 or not math.isfinite(transition_exponent):
        raise ValueError("transition exponent must be finite and positive")

    boundary = ordered_boundary_loop(faces, vertices_3d.shape[0])
    boundary_uv = uv[boundary]
    if float(_signed_polygon_area(boundary_uv).detach()) < 0.0:
        boundary = torch.flip(boundary, dims=(0,))
        boundary_uv = uv[boundary]

    center = boundary_uv.mean(dim=0)
    edges = torch.roll(boundary_uv, shifts=-1, dims=0) - boundary_uv
    edge_lengths = torch.linalg.vector_norm(edges, dim=-1)
    cumulative = torch.cat(
        (
            torch.zeros(1, dtype=uv.dtype, device=uv.device),
            torch.cumsum(edge_lengths[:-1], dim=0),
        )
    )
    fractions = cumulative / edge_lengths.sum().clamp_min(1.0e-15)
    start_direction = boundary_uv[0] - center
    start_angle = torch.atan2(start_direction[1], start_direction[0])
    angles = start_angle + 2.0 * math.pi * fractions
    radius = scale * torch.linalg.vector_norm(boundary_uv - center, dim=-1).amax()
    outer_uv = center + radius * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )

    boundary_count = boundary.numel()
    first_scaffold = vertices_3d.shape[0]
    previous = boundary
    ring_uvs: list[Tensor] = []
    ring_vertices: list[Tensor] = []
    ring_indices: list[Tensor] = []
    annulus_faces: list[list[int]] = []
    for ring_index in range(1, rings + 1):
        fraction = float((ring_index / rings) ** transition_exponent)
        current_uv = (1.0 - fraction) * boundary_uv + fraction * outer_uv
        first_current = first_scaffold + (ring_index - 1) * boundary_count
        current = torch.arange(
            first_current,
            first_current + boundary_count,
            dtype=torch.long,
            device=faces.device,
        )
        ring_uvs.append(current_uv)
        ring_vertices.append(vertices_3d[boundary].clone())
        ring_indices.append(current)
        for offset in range(boundary_count):
            following = (offset + 1) % boundary_count
            inner_a = int(previous[offset])
            inner_b = int(previous[following])
            outer_a = int(current[offset])
            outer_b = int(current[following])
            annulus_faces.extend(
                ([inner_a, outer_a, outer_b], [inner_a, outer_b, inner_b])
            )
        previous = current

    extended_faces = torch.cat(
        (faces, torch.tensor(annulus_faces, dtype=torch.long, device=faces.device)),
        dim=0,
    )
    extended_uv = torch.cat((uv, *ring_uvs), dim=0)
    extended_vertices = torch.cat((vertices_3d, *ring_vertices), dim=0)
    areas = signed_double_areas(extended_uv, extended_faces)
    if bool(torch.any(areas.detach() <= 0.0)):
        raise ValueError("constructed multiring scaffold is not positively oriented")
    if count_proper_edge_intersections(extended_uv.detach(), extended_faces) != 0:
        raise ValueError("constructed multiring scaffold has a proper edge intersection")
    return ScaffoldMesh(
        extended_vertices,
        extended_faces,
        extended_uv,
        vertices_3d.shape[0],
        faces.shape[0],
        boundary,
        ring_indices[-1],
    )
