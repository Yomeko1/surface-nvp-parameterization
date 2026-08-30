"""Trainable graph-conditioned coupling layers for the standalone PL-NVP prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .mesh_coupling import (
    boundary_vertices,
    greedy_vertex_coloring,
    signed_double_areas,
    vertex_adjacency,
    vertex_kernel,
)
from .radial_polytope import analytic_center, from_polytope, ray_radius, to_polytope


class GraphConditioner(nn.Module):
    """Small MLP fed only by frozen-neighbor UV features and fixed 3D metadata."""

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        # Every layer starts as the identity. Earlier layers still receive
        # gradients because the final linear weights are trainable.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


@dataclass
class CouplingDiagnostics:
    q_values: Tensor
    halfplane_slacks: Tensor
    vertex_ids: Tensor
    layer_ids: Tensor


class MeshCouplingFlow(nn.Module):
    """A sequence of color coupling layers with hard one-ring legality."""

    def __init__(
        self,
        vertices_3d: Tensor,
        faces: Tensor,
        initial_uv: Tensor,
        *,
        cycles: int = 2,
        hidden_dim: int = 32,
        feature_set: str = "basic",
        max_log_scale: float = 0.08,
        max_shift_fraction: float = 0.04,
        center_iterations: int = 12,
    ) -> None:
        super().__init__()
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if feature_set not in {"basic", "local-geometry"}:
            raise ValueError("feature_set must be 'basic' or 'local-geometry'")
        self.register_buffer("vertices_3d", vertices_3d)
        self.register_buffer("faces", faces)
        self.register_buffer("initial_uv", initial_uv)
        colors = greedy_vertex_coloring(faces, vertices_3d.shape[0])
        boundary = boundary_vertices(faces, vertices_3d.shape[0])
        self.register_buffer("colors", colors)
        self.register_buffer("boundary", boundary)

        xyz_center = vertices_3d.mean(dim=0)
        xyz_scale = torch.linalg.vector_norm(vertices_3d - xyz_center, dim=-1).amax().clamp_min(1.0e-12)
        self.register_buffer("normalized_xyz", (vertices_3d - xyz_center) / xyz_scale)
        uv_center = initial_uv.mean(dim=0)
        uv_scale = torch.linalg.vector_norm(initial_uv - uv_center, dim=-1).amax().clamp_min(1.0e-12)
        self.register_buffer("uv_center", uv_center)
        self.register_buffer("uv_scale", uv_scale)

        self.cycles = cycles
        self.feature_set = feature_set
        self.color_count = int(colors.max().item()) + 1
        self.max_log_scale = float(max_log_scale)
        self.max_shift_fraction = float(max_shift_fraction)
        self.center_iterations = int(center_iterations)
        self._adjacency = vertex_adjacency(faces, vertices_3d.shape[0])
        self.register_buffer("local_geometry_features", self._local_geometry_features())
        self._active_sets: list[Tensor] = []
        for _ in range(cycles):
            for color in range(self.color_count):
                active = torch.nonzero((colors == color) & ~boundary, as_tuple=False).flatten()
                self._active_sets.append(active)
        feature_dim = 8 if feature_set == "basic" else 14
        self.conditioners = nn.ModuleList(
            GraphConditioner(feature_dim=feature_dim, hidden_dim=hidden_dim)
            for active in self._active_sets
            if active.numel()
        )
        self._nonempty_layer_indices = [i for i, active in enumerate(self._active_sets) if active.numel()]
        if len(self.conditioners) != len(self._nonempty_layer_indices):
            raise RuntimeError("conditioner construction mismatch")

    def _local_geometry_features(self) -> Tensor:
        """Fixed scale/area statistics; safe for an explicit coupling inverse."""
        triangles = self.normalized_xyz[self.faces]
        face_areas = 0.5 * torch.linalg.vector_norm(
            torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=-1,
            ),
            dim=-1,
        )
        rows: list[Tensor] = []
        for vertex in range(self.vertices_3d.shape[0]):
            neighbor_ids = torch.tensor(
                sorted(self._adjacency[vertex]),
                dtype=torch.long,
                device=self.vertices_3d.device,
            )
            lengths = torch.linalg.vector_norm(
                self.normalized_xyz[neighbor_ids] - self.normalized_xyz[vertex], dim=-1
            )
            incident = torch.any(self.faces == vertex, dim=1)
            incident_areas = face_areas[incident]
            rows.append(
                torch.stack(
                    (
                        lengths.mean(),
                        lengths.std(unbiased=False),
                        lengths.amin(),
                        lengths.amax(),
                        incident_areas.mean(),
                        incident_areas.sum(),
                    )
                )
            )
        return torch.stack(rows)

    def _features(self, uv: Tensor, active: Tensor) -> Tensor:
        features: list[Tensor] = []
        for vertex in active.detach().cpu().tolist():
            neighbor_ids = torch.tensor(
                sorted(self._adjacency[int(vertex)]), dtype=torch.long, device=uv.device
            )
            neighbors = uv[neighbor_ids]
            normalized_neighbors = (neighbors - self.uv_center) / self.uv_scale
            mean = normalized_neighbors.mean(dim=0)
            std = torch.sqrt(normalized_neighbors.var(dim=0, unbiased=False) + 1.0e-12)
            degree = torch.as_tensor(
                [neighbors.shape[0] / 12.0], dtype=uv.dtype, device=uv.device
            )
            parts = [self.normalized_xyz[int(vertex)], mean, std, degree]
            if self.feature_set == "local-geometry":
                parts.append(self.local_geometry_features[int(vertex)])
            features.append(torch.cat(parts))
        return torch.stack(features)

    def _sublayer(
        self,
        uv: Tensor,
        active: Tensor,
        conditioner: GraphConditioner,
        *,
        inverse: bool,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        raw = conditioner(self._features(uv, active))
        log_scale = self.max_log_scale * torch.tanh(raw[:, :2])
        shift = self.max_shift_fraction * self.uv_scale * torch.tanh(raw[:, 2:])

        source = uv
        result = uv.clone()
        q_values: list[Tensor] = []
        slacks: list[Tensor] = []
        for local_index, vertex_value in enumerate(active.detach().cpu().tolist()):
            vertex = int(vertex_value)
            A, b = vertex_kernel(source, self.faces, vertex)
            center = analytic_center(A, b, iterations=self.center_iterations)
            latent = from_polytope(source[vertex], A, b, center)
            if inverse:
                transformed = (latent - shift[local_index]) * torch.exp(-log_scale[local_index])
            else:
                transformed = latent * torch.exp(log_scale[local_index]) + shift[local_index]
            mapped = to_polytope(transformed, A, b, center)
            result[vertex] = mapped

            delta = mapped - center
            radius = torch.linalg.vector_norm(delta)
            tiny = 32.0 * torch.finfo(uv.dtype).eps
            nonzero = radius > tiny
            safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
            direction = delta / safe_radius
            fallback = torch.tensor([1.0, 0.0], dtype=uv.dtype, device=uv.device)
            direction = torch.where(nonzero, direction, fallback)
            boundary_radius = ray_radius(direction, A, b, center)
            q_values.append(torch.where(nonzero, radius / boundary_radius, torch.zeros_like(radius)))
            slacks.append((b - A @ mapped).amin())
        return result, q_values, slacks

    def forward(
        self,
        uv: Tensor | None = None,
        *,
        inverse: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, CouplingDiagnostics]:
        result = self.initial_uv if uv is None else uv
        layers = list(zip(self._nonempty_layer_indices, self.conditioners, strict=True))
        if inverse:
            layers.reverse()
        all_q: list[Tensor] = []
        all_slacks: list[Tensor] = []
        all_vertex_ids: list[Tensor] = []
        all_layer_ids: list[Tensor] = []
        for active_index, conditioner in layers:
            result, q_values, slacks = self._sublayer(
                result,
                self._active_sets[active_index],
                conditioner,
                inverse=inverse,
            )
            all_q.extend(q_values)
            all_slacks.extend(slacks)
            active = self._active_sets[active_index]
            all_vertex_ids.append(active)
            all_layer_ids.append(torch.full_like(active, active_index))
        if return_diagnostics:
            return result, CouplingDiagnostics(
                torch.stack(all_q),
                torch.stack(all_slacks),
                torch.cat(all_vertex_ids),
                torch.cat(all_layer_ids),
            )
        return result

    def minimum_double_area(self, uv: Tensor) -> Tensor:
        return signed_double_areas(uv, self.faces).amin()
