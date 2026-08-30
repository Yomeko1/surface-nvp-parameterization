"""Small mesh-coupling prototype built on the radial polytope map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .radial_polytope import analytic_center, from_polytope, to_polytope


def signed_double_areas(vertices: Tensor, faces: Tensor) -> Tensor:
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def vertex_adjacency(faces: Tensor, vertex_count: int) -> list[set[int]]:
    adjacency = [set() for _ in range(vertex_count)]
    for face in faces.detach().cpu().tolist():
        for offset in range(3):
            a, b = face[offset], face[(offset + 1) % 3]
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


def greedy_vertex_coloring(faces: Tensor, vertex_count: int) -> Tensor:
    """Deterministic largest-degree-first proper coloring."""
    adjacency = vertex_adjacency(faces, vertex_count)
    order = sorted(range(vertex_count), key=lambda i: (-len(adjacency[i]), i))
    colors = [-1] * vertex_count
    for vertex in order:
        forbidden = {colors[n] for n in adjacency[vertex] if colors[n] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
    return torch.tensor(colors, dtype=torch.long, device=faces.device)


def boundary_vertices(faces: Tensor, vertex_count: int) -> Tensor:
    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces.detach().cpu().tolist():
        for offset in range(3):
            a, b = face[offset], face[(offset + 1) % 3]
            edge = (min(a, b), max(a, b))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    result = torch.zeros(vertex_count, dtype=torch.bool, device=faces.device)
    for (a, b), count in edge_counts.items():
        if count == 1:
            result[a] = True
            result[b] = True
    return result


def _incident_oriented_neighbor_pairs(faces: Tensor, vertex: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for a, b, c in faces.detach().cpu().tolist():
        if vertex == a:
            pairs.append((b, c))
        elif vertex == b:
            pairs.append((c, a))
        elif vertex == c:
            pairs.append((a, b))
    if not pairs:
        raise ValueError(f"vertex {vertex} has no incident faces")
    return pairs


def vertex_kernel(vertices: Tensor, faces: Tensor, vertex: int) -> tuple[Tensor, Tensor]:
    """Half-planes preserving all incident face orientations for one vertex."""
    rows: list[Tensor] = []
    offsets: list[Tensor] = []
    for j, k in _incident_oriented_neighbor_pairs(faces, vertex):
        edge = vertices[k] - vertices[j]
        normal = torch.stack((edge[1], -edge[0]))
        length = torch.linalg.vector_norm(normal)
        if float(length.detach()) <= torch.finfo(vertices.dtype).eps:
            raise ValueError("degenerate neighbor edge")
        normal = normal / length
        rows.append(normal)
        offsets.append(torch.dot(normal, vertices[j]))
    return torch.stack(rows), torch.stack(offsets)


def _conditioner(neighbors: Tensor) -> tuple[Tensor, Tensor]:
    """A deterministic graph conditioner used only by the standalone demo."""
    mean = neighbors.mean(dim=0)
    centered = neighbors - mean
    spread = torch.sqrt(torch.mean(centered.square(), dim=0) + 1.0e-12)
    scale = 0.30 * torch.tanh(torch.stack((mean[0] - mean[1], spread[1] - spread[0])))
    shift = 0.55 * torch.stack(
        (
            torch.sin(2.7 * mean[0] + 1.3 * spread[1]),
            torch.cos(2.1 * mean[1] - 1.7 * spread[0]),
        )
    )
    return scale, shift


def coupling_sublayer(
    vertices: Tensor,
    faces: Tensor,
    active_vertices: Tensor,
    *,
    inverse: bool = False,
) -> Tensor:
    """Update an independent vertex set with an explicitly invertible coupling."""
    active_list = [int(i) for i in active_vertices.detach().cpu().tolist()]
    active_set = set(active_list)
    adjacency = vertex_adjacency(faces, vertices.shape[0])
    for vertex in active_list:
        if adjacency[vertex].intersection(active_set):
            raise ValueError("active vertices must form an independent set")

    source = vertices
    result = vertices.clone()
    for vertex in active_list:
        A, b = vertex_kernel(source, faces, vertex)
        center = analytic_center(A, b)
        neighbors = source[torch.tensor(sorted(adjacency[vertex]), device=vertices.device)]
        log_scale, shift = _conditioner(neighbors)
        latent = from_polytope(source[vertex], A, b, center)
        if inverse:
            transformed = (latent - shift) * torch.exp(-log_scale)
        else:
            transformed = latent * torch.exp(log_scale) + shift
        result[vertex] = to_polytope(transformed, A, b, center)
    return result


def coupling_cycle(vertices: Tensor, faces: Tensor, *, inverse: bool = False) -> Tensor:
    """Apply all interior color classes, or invert them in reverse order."""
    colors = greedy_vertex_coloring(faces, vertices.shape[0])
    boundary = boundary_vertices(faces, vertices.shape[0])
    color_ids = list(range(int(colors.max().item()) + 1))
    if inverse:
        color_ids.reverse()

    result = vertices
    for color in color_ids:
        active = torch.nonzero((colors == color) & ~boundary, as_tuple=False).flatten()
        if active.numel():
            result = coupling_sublayer(result, faces, active, inverse=inverse)
    return result


def make_grid_triangulation(nx: int = 6, ny: int = 6, *, dtype=torch.float64) -> tuple[Tensor, Tensor]:
    if nx < 3 or ny < 3:
        raise ValueError("grid must contain interior vertices")
    vertices = torch.tensor(
        [(i / (nx - 1), j / (ny - 1)) for j in range(ny) for i in range(nx)],
        dtype=dtype,
    )
    faces: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            d = (j + 1) * nx + i
            c = d + 1
            if (i + j) % 2 == 0:
                faces.extend(((a, b, c), (a, c, d)))
            else:
                faces.extend(((a, b, d), (b, c, d)))
    return vertices, torch.tensor(faces, dtype=torch.long)


def count_proper_edge_intersections(
    vertices: Tensor,
    faces: Tensor,
    tolerance: float = 1.0e-12,
    pair_batch_size: int = 262_144,
) -> int:
    """Count strict crossings between non-adjacent edges in vectorized batches."""
    edges: set[tuple[int, int]] = set()
    for face in faces.detach().cpu().tolist():
        for offset in range(3):
            a, b = face[offset], face[(offset + 1) % 3]
            edges.add((min(a, b), max(a, b)))
    edge_tensor = torch.tensor(sorted(edges), dtype=torch.long, device=vertices.device)
    pair_indices = torch.triu_indices(
        edge_tensor.shape[0], edge_tensor.shape[0], offset=1, device=vertices.device
    )
    intersections = 0
    for start in range(0, pair_indices.shape[1], pair_batch_size):
        pairs = pair_indices[:, start : start + pair_batch_size]
        first = edge_tensor[pairs[0]]
        second = edge_tensor[pairs[1]]
        disjoint = (
            (first[:, 0] != second[:, 0])
            & (first[:, 0] != second[:, 1])
            & (first[:, 1] != second[:, 0])
            & (first[:, 1] != second[:, 1])
        )
        if not bool(torch.any(disjoint)):
            continue
        first = first[disjoint]
        second = second[disjoint]
        a, b = vertices[first[:, 0]], vertices[first[:, 1]]
        c, d = vertices[second[:, 0]], vertices[second[:, 1]]

        def orientation(p: Tensor, q: Tensor, r: Tensor) -> Tensor:
            return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (
                q[:, 1] - p[:, 1]
            ) * (r[:, 0] - p[:, 0])

        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        crossings = (o1 * o2 < -tolerance) & (o3 * o4 < -tolerance)
        intersections += int(torch.count_nonzero(crossings).detach())
    return intersections


@dataclass(frozen=True)
class EmbeddingDiagnostics:
    minimum_double_area: float
    flipped_faces: int
    proper_edge_intersections: int


def diagnose_embedding(vertices: Tensor, faces: Tensor) -> EmbeddingDiagnostics:
    areas = signed_double_areas(vertices, faces)
    return EmbeddingDiagnostics(
        minimum_double_area=float(areas.detach().min()),
        flipped_faces=int(torch.count_nonzero(areas.detach() <= 0.0)),
        proper_edge_intersections=count_proper_edge_intersections(vertices, faces),
    )
