"""Batched same-color implementation of the trainable mesh coupling flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .trainable_coupling import CouplingDiagnostics, GraphConditioner, MeshCouplingFlow


@dataclass(frozen=True)
class LayerTopology:
    active: Tensor
    neighbor_ids: Tensor
    neighbor_mask: Tensor
    pair_j: Tensor
    pair_k: Tensor
    pair_mask: Tensor

    def on(self, device: torch.device) -> "LayerTopology":
        if self.active.device == device:
            return self
        return LayerTopology(
            active=self.active.to(device),
            neighbor_ids=self.neighbor_ids.to(device),
            neighbor_mask=self.neighbor_mask.to(device),
            pair_j=self.pair_j.to(device),
            pair_k=self.pair_k.to(device),
            pair_mask=self.pair_mask.to(device),
        )


def _oriented_pairs_by_vertex(faces: Tensor, vertex_count: int) -> list[list[tuple[int, int]]]:
    pairs: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces.detach().cpu().tolist():
        pairs[a].append((b, c))
        pairs[b].append((c, a))
        pairs[c].append((a, b))
    return pairs


def _pad_index_lists(index_lists: list[list[int]], device: torch.device) -> tuple[Tensor, Tensor]:
    width = max(map(len, index_lists))
    indices = torch.zeros((len(index_lists), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(index_lists), width), dtype=torch.bool, device=device)
    for row, values in enumerate(index_lists):
        indices[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = True
    return indices, mask


def build_layer_topology(
    active: Tensor,
    adjacency: list[set[int]],
    oriented_pairs: list[list[tuple[int, int]]],
) -> LayerTopology:
    active_list = [int(value) for value in active.detach().cpu().tolist()]
    neighbor_lists = [sorted(adjacency[vertex]) for vertex in active_list]
    neighbor_ids, neighbor_mask = _pad_index_lists(neighbor_lists, active.device)

    pair_lists = [oriented_pairs[vertex] for vertex in active_list]
    width = max(map(len, pair_lists))
    pair_j = torch.zeros((len(active_list), width), dtype=torch.long, device=active.device)
    pair_k = torch.zeros_like(pair_j)
    pair_mask = torch.zeros((len(active_list), width), dtype=torch.bool, device=active.device)
    for row, values in enumerate(pair_lists):
        if values:
            pair_j[row, : len(values)] = torch.tensor([value[0] for value in values], device=active.device)
            pair_k[row, : len(values)] = torch.tensor([value[1] for value in values], device=active.device)
            pair_mask[row, : len(values)] = True
    return LayerTopology(active, neighbor_ids, neighbor_mask, pair_j, pair_k, pair_mask)


def batched_vertex_kernels(uv: Tensor, topology: LayerTopology) -> tuple[Tensor, Tensor, Tensor]:
    topology = topology.on(uv.device)
    point_j = uv[topology.pair_j]
    point_k = uv[topology.pair_k]
    edges = point_k - point_j
    normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1)
    lengths = torch.linalg.vector_norm(normals, dim=-1)
    safe_lengths = torch.where(topology.pair_mask, lengths.clamp_min(1.0e-15), torch.ones_like(lengths))
    A = normals / safe_lengths[..., None]
    A = torch.where(topology.pair_mask[..., None], A, torch.zeros_like(A))
    b = torch.sum(A * point_j, dim=-1)
    return A, b, topology.pair_mask


def batched_analytic_center(
    A: Tensor,
    b: Tensor,
    mask: Tensor,
    *,
    iterations: int = 16,
) -> Tensor:
    """Damped Newton analytic centers for a batch of padded polygons."""
    center = batched_polygon_vertex_mean(A, b, mask)
    for _ in range(iterations):
        slack = b - torch.einsum("nmi,ni->nm", A, center)
        safe_slack = torch.where(mask, slack, torch.ones_like(slack))
        inverse_slack = torch.where(mask, safe_slack.reciprocal(), torch.zeros_like(slack))
        gradient = torch.einsum("nmi,nm->ni", A, inverse_slack)
        hessian = torch.einsum("nmi,nmj,nm->nij", A, A, inverse_slack.square())
        newton_step = torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
        # A strictly positive floor avoids the infinite derivative of sqrt at
        # an exactly converged/symmetric analytic center.
        decrement = torch.sqrt(torch.sum(gradient * newton_step, dim=-1).clamp_min(1.0e-24))
        step_size = 1.0 / (1.0 + decrement)

        # Enforce a strict feasible fraction even under floating-point error.
        a_step = torch.einsum("nmi,ni->nm", A, newton_step)
        limiting = mask & (a_step < 0.0)
        feasible_steps = torch.where(
            limiting,
            0.99 * safe_slack / (-a_step).clamp_min(1.0e-30),
            torch.full_like(slack, torch.inf),
        )
        maximum_step = feasible_steps.amin(dim=-1)
        step_size = torch.minimum(step_size, maximum_step)
        center = center - step_size[:, None] * newton_step
    return center


def batched_polygon_vertex_mean(
    A: Tensor,
    b: Tensor,
    mask: Tensor,
    *,
    parallel_tolerance: float = 1.0e-12,
    feasibility_tolerance: float = 1.0e-9,
) -> Tensor:
    """Vectorized mean of feasible pairwise boundary-line intersections."""
    constraint_count = A.shape[1]
    pairs = torch.triu_indices(
        constraint_count, constraint_count, offset=1, device=A.device
    )
    ai = A[:, pairs[0]]
    aj = A[:, pairs[1]]
    bi = b[:, pairs[0]]
    bj = b[:, pairs[1]]
    determinant = ai[..., 0] * aj[..., 1] - ai[..., 1] * aj[..., 0]
    pair_valid = (
        mask[:, pairs[0]]
        & mask[:, pairs[1]]
        & (torch.abs(determinant) > parallel_tolerance)
    )
    safe_determinant = torch.where(pair_valid, determinant, torch.ones_like(determinant))
    points = torch.stack(
        (
            (bi * aj[..., 1] - ai[..., 1] * bj) / safe_determinant,
            (ai[..., 0] * bj - bi * aj[..., 0]) / safe_determinant,
        ),
        dim=-1,
    )
    violations = torch.einsum("nmi,npi->npm", A, points) - b[:, None, :]
    constraints_satisfied = (violations <= feasibility_tolerance) | ~mask[:, None, :]
    feasible = pair_valid & torch.all(constraints_satisfied, dim=-1)
    count = feasible.sum(dim=-1)
    if bool(torch.any(count.detach() < 3)):
        raise ValueError("a padded half-plane set does not define a non-degenerate polygon")
    center = torch.sum(points * feasible[..., None].to(A.dtype), dim=1) / count[:, None]
    slack = b - torch.einsum("nmi,ni->nm", A, center)
    minimum_slack = torch.where(mask, slack, torch.full_like(slack, torch.inf)).amin(dim=-1)
    if bool(torch.any(minimum_slack.detach() <= 0.0)):
        raise ValueError("batched polygon vertex mean is not strictly interior")
    return center


def batched_ray_radius(
    directions: Tensor,
    A: Tensor,
    b: Tensor,
    center: Tensor,
    mask: Tensor,
) -> Tensor:
    slack = b - torch.einsum("nmi,ni->nm", A, center)
    denominators = torch.einsum("nmi,ni->nm", A, directions)
    valid = mask & (denominators > 32.0 * torch.finfo(A.dtype).eps)
    candidates = torch.where(
        valid,
        slack / torch.where(valid, denominators, torch.ones_like(denominators)),
        torch.full_like(denominators, torch.inf),
    )
    radius = candidates.amin(dim=-1)
    if bool(torch.any(~torch.isfinite(radius.detach()))):
        raise ValueError("at least one padded polygon is unbounded in a requested direction")
    return radius


def _radius_direction(delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    radius = torch.linalg.vector_norm(delta, dim=-1)
    tiny = 32.0 * torch.finfo(delta.dtype).eps
    nonzero = radius > tiny
    safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
    direction = delta / safe_radius[:, None]
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    direction = torch.where(nonzero[:, None], direction, fallback)
    return radius, direction, nonzero


def batched_from_polytope(points: Tensor, A: Tensor, b: Tensor, center: Tensor, mask: Tensor) -> Tensor:
    delta = points - center
    radius, direction, nonzero = _radius_direction(delta)
    boundary_radius = batched_ray_radius(direction, A, b, center, mask)
    fraction = radius / boundary_radius
    if bool(torch.any(fraction.detach() >= 1.0)):
        raise ValueError("a batched input point lies on or outside its polygon")
    unsquashed = fraction / (1.0 - fraction)
    latent = center + (boundary_radius * unsquashed)[:, None] * direction
    center_limit = center + delta
    return torch.where(nonzero[:, None], latent, center_limit)


def batched_to_polytope(latent: Tensor, A: Tensor, b: Tensor, center: Tensor, mask: Tensor) -> Tensor:
    delta = latent - center
    radius, direction, nonzero = _radius_direction(delta)
    boundary_radius = batched_ray_radius(direction, A, b, center, mask)
    relative_radius = radius / boundary_radius
    squashed = relative_radius / (1.0 + relative_radius)
    mapped = center + (boundary_radius * squashed)[:, None] * direction
    center_limit = center + delta
    return torch.where(nonzero[:, None], mapped, center_limit)


class BatchedMeshCouplingFlow(MeshCouplingFlow):
    """Same mathematical layer as MeshCouplingFlow, batched within each color."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        oriented_pairs = _oriented_pairs_by_vertex(self.faces, self.vertices_3d.shape[0])
        unique: dict[tuple[int, ...], LayerTopology] = {}
        for active in self._active_sets:
            if not active.numel():
                continue
            key = tuple(int(value) for value in active.detach().cpu().tolist())
            if key not in unique:
                unique[key] = build_layer_topology(active, self._adjacency, oriented_pairs)
        self._topology_by_key = unique

    def _features_batched(self, uv: Tensor, topology: LayerTopology) -> Tensor:
        topology = topology.on(uv.device)
        neighbors = uv[topology.neighbor_ids]
        normalized = (neighbors - self.uv_center) / self.uv_scale
        weights = topology.neighbor_mask.to(uv.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = torch.sum(normalized * weights[..., None], dim=1) / count
        centered = normalized - mean[:, None, :]
        variance = torch.sum(centered.square() * weights[..., None], dim=1) / count
        std = torch.sqrt(variance + 1.0e-12)
        degree = count / 12.0
        parts = [self.normalized_xyz[topology.active], mean, std, degree]
        if self.feature_set == "local-geometry":
            parts.append(self.local_geometry_features[topology.active])
        return torch.cat(parts, dim=-1)

    def _sublayer(
        self,
        uv: Tensor,
        active: Tensor,
        conditioner: GraphConditioner,
        *,
        inverse: bool,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        key = tuple(int(value) for value in active.detach().cpu().tolist())
        topology = self._topology_by_key[key].on(uv.device)
        raw = conditioner(self._features_batched(uv, topology))
        log_scale = self.max_log_scale * torch.tanh(raw[:, :2])
        shift = self.max_shift_fraction * self.uv_scale * torch.tanh(raw[:, 2:])

        A, b, mask = batched_vertex_kernels(uv, topology)
        center = batched_analytic_center(A, b, mask, iterations=self.center_iterations)
        latent = batched_from_polytope(uv[topology.active], A, b, center, mask)
        if inverse:
            transformed = (latent - shift) * torch.exp(-log_scale)
        else:
            transformed = latent * torch.exp(log_scale) + shift
        mapped = batched_to_polytope(transformed, A, b, center, mask)
        result = uv.index_copy(0, topology.active, mapped)

        delta = mapped - center
        radius, direction, nonzero = _radius_direction(delta)
        boundary_radius = batched_ray_radius(direction, A, b, center, mask)
        q = torch.where(nonzero, radius / boundary_radius, torch.zeros_like(radius))
        slack = b - torch.einsum("nmi,ni->nm", A, mapped)
        slack = torch.where(mask, slack, torch.full_like(slack, torch.inf)).amin(dim=-1)
        return result, list(q.unbind()), list(slack.unbind())
