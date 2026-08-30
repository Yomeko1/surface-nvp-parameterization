"""One-ring outer scaffold for freeing the original mesh boundary."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .mesh_coupling import count_proper_edge_intersections, signed_double_areas


@dataclass(frozen=True)
class ScaffoldMesh:
    vertices_3d: Tensor
    faces: Tensor
    uv: Tensor
    original_vertex_count: int
    original_face_count: int
    original_boundary: Tensor
    outer_boundary: Tensor


def ordered_boundary_loop(faces: Tensor, vertex_count: int) -> Tensor:
    edge_counts = Counter(
        tuple(sorted((int(face[offset]), int(face[(offset + 1) % 3]))))
        for face in faces.detach().cpu().tolist()
        for offset in range(3)
    )
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("scaffold requires one manifold boundary loop")

    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    while True:
        candidates = adjacency[current]
        following = candidates[0] if candidates[0] != previous else candidates[1]
        if following == start:
            break
        if following in loop:
            raise ValueError("boundary edges contain more than one loop")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacency):
        raise ValueError("scaffold currently supports exactly one boundary loop")
    return torch.tensor(loop, dtype=torch.long, device=faces.device)


def _signed_polygon_area(points: Tensor) -> Tensor:
    following = torch.roll(points, shifts=-1, dims=0)
    return 0.5 * torch.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0])


def build_outer_scaffold(
    vertices_3d: Tensor,
    faces: Tensor,
    uv: Tensor,
    *,
    scale: float = 1.1,
) -> ScaffoldMesh:
    """Add a convex fixed outer ring with one vertex per source boundary vertex."""
    if scale <= 1.0:
        raise ValueError("scaffold scale must be greater than one")
    boundary = ordered_boundary_loop(faces, vertices_3d.shape[0])
    boundary_uv = uv[boundary]
    if float(_signed_polygon_area(boundary_uv).detach()) < 0.0:
        boundary = torch.flip(boundary, dims=(0,))
        boundary_uv = uv[boundary]

    center = boundary_uv.mean(dim=0)
    edge_lengths = torch.linalg.vector_norm(
        torch.roll(boundary_uv, shifts=-1, dims=0) - boundary_uv, dim=-1
    )
    cumulative = torch.cat(
        (torch.zeros(1, dtype=uv.dtype, device=uv.device), torch.cumsum(edge_lengths[:-1], dim=0))
    )
    fractions = cumulative / edge_lengths.sum().clamp_min(1.0e-15)
    start_direction = boundary_uv[0] - center
    start_angle = torch.atan2(start_direction[1], start_direction[0])
    angles = start_angle + (2.0 * math.pi) * fractions
    radius = scale * torch.linalg.vector_norm(boundary_uv - center, dim=-1).amax()
    outer_uv = center + radius * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    # Outer vertices are fixed and never enter the loss. Duplicating the
    # corresponding boundary XYZ avoids inventing an artificial 3D surface.
    outer_vertices_3d = vertices_3d[boundary].clone()
    first_outer = vertices_3d.shape[0]
    outer = torch.arange(
        first_outer,
        first_outer + boundary.shape[0],
        dtype=torch.long,
        device=faces.device,
    )
    annulus_faces: list[list[int]] = []
    for offset in range(boundary.shape[0]):
        following = (offset + 1) % boundary.shape[0]
        inner_a = int(boundary[offset])
        inner_b = int(boundary[following])
        outer_a = int(outer[offset])
        outer_b = int(outer[following])
        annulus_faces.extend(([inner_a, outer_a, outer_b], [inner_a, outer_b, inner_b]))
    annulus = torch.tensor(annulus_faces, dtype=torch.long, device=faces.device)
    extended_faces = torch.cat((faces, annulus), dim=0)
    extended_uv = torch.cat((uv, outer_uv), dim=0)
    extended_vertices = torch.cat((vertices_3d, outer_vertices_3d), dim=0)

    areas = signed_double_areas(extended_uv, extended_faces)
    if bool(torch.any(areas.detach() <= 0.0)):
        raise ValueError("constructed scaffold is not positively oriented")
    if count_proper_edge_intersections(extended_uv.detach(), extended_faces) != 0:
        raise ValueError("constructed scaffold has a proper edge intersection")
    return ScaffoldMesh(
        extended_vertices,
        extended_faces,
        extended_uv,
        vertices_3d.shape[0],
        faces.shape[0],
        boundary,
        outer,
    )
