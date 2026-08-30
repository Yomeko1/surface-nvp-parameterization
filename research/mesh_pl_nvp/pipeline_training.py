"""Pipeline-style trainer for the isolated mesh-aligned PL-NVP prototype.

This module intentionally lives under ``research``.  It reuses the v2.4
distortion implementation while keeping the topology-dependent model and its
no-rollback training loop outside the production package.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from surface_nvp.losses.distortion import symmetric_dirichlet_loss

from .adaptive_lr import AdaptivePlateauController
from .batched_coupling import BatchedMeshCouplingFlow
from .mesh_coupling import signed_double_areas
from .scaffold import ScaffoldMesh, build_outer_scaffold, ordered_boundary_loop


DTYPE = torch.float64


@dataclass
class MeshPLTrainingResult:
    uv: np.ndarray
    model: BatchedMeshCouplingFlow
    history: list[dict[str, Any]]
    info: dict[str, Any]
    final_diagnostics: Any
    model_initial_uv: Tensor
    model_final_uv: Tensor
    model_faces: Tensor
    scaffold: ScaffoldMesh | None


def validate_disk_topology(faces: np.ndarray, vertex_count: int) -> dict[str, int]:
    """Require the connected, manifold, single-boundary disk used by the proof."""
    counts = Counter(
        tuple(sorted((int(face[offset]), int(face[(offset + 1) % 3]))))
        for face in faces
        for offset in range(3)
    )
    nonmanifold = sum(count > 2 for count in counts.values())
    if nonmanifold:
        raise ValueError(f"mesh has {nonmanifold} non-manifold edges")

    adjacency = [set() for _ in range(vertex_count)]
    for a, b in counts:
        adjacency[a].add(b)
        adjacency[b].add(a)
    if vertex_count:
        seen = {0}
        queue = deque([0])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    else:
        seen = set()
    if len(seen) != vertex_count:
        raise ValueError("mesh must have exactly one connected component")

    faces_t = torch.as_tensor(faces, dtype=torch.long)
    boundary = ordered_boundary_loop(faces_t, vertex_count)
    euler = vertex_count - len(counts) + len(faces)
    if euler != 1:
        raise ValueError(f"mesh must be a topological disk (Euler characteristic is {euler}, not 1)")
    return {
        "vertex_count": int(vertex_count),
        "face_count": int(len(faces)),
        "edge_count": int(len(counts)),
        "boundary_vertex_count": int(boundary.numel()),
        "euler_characteristic": int(euler),
    }


def train_mesh_pl_nvp(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv0: np.ndarray,
    config: dict[str, Any],
    *,
    initial_model_state: str | None = None,
) -> MeshPLTrainingResult:
    """Train PL-NVP using final-state optimization and no rollback mechanism."""
    topology = validate_disk_topology(faces, len(vertices))
    model_config = config["model"]
    train_config = config["train"]
    scaffold_config = config["scaffold"]
    device = torch.device(train_config["device"])

    torch.manual_seed(int(train_config["seed"]))
    np.random.seed(int(train_config["seed"]))
    vertices_t = torch.as_tensor(vertices, dtype=DTYPE, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    initial_uv_t = torch.as_tensor(uv0, dtype=DTYPE, device=device)
    original_vertex_count = vertices_t.shape[0]

    scaffold = None
    model_vertices = vertices_t
    model_faces = faces_t
    model_initial_uv = initial_uv_t
    if scaffold_config["enabled"]:
        scaffold = build_outer_scaffold(
            vertices_t,
            faces_t,
            initial_uv_t,
            scale=float(scaffold_config["scale"]),
        )
        model_vertices = scaffold.vertices_3d
        model_faces = scaffold.faces
        model_initial_uv = scaffold.uv

    model = BatchedMeshCouplingFlow(
        model_vertices,
        model_faces,
        model_initial_uv,
        cycles=int(model_config["cycles"]),
        hidden_dim=int(model_config["hidden_dim"]),
        feature_set=model_config["conditioner_features"],
        max_log_scale=float(model_config["max_log_scale"]),
        max_shift_fraction=float(model_config["max_shift_fraction"]),
        center_iterations=int(model_config["center_iterations"]),
    ).to(device=device, dtype=DTYPE)
    if initial_model_state is not None:
        state = torch.load(initial_model_state, map_location=device, weights_only=True)
        model.load_state_dict(state)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config["lr"]))
    with torch.no_grad():
        start_uv = model()
        initial_original_area = float(
            signed_double_areas(start_uv[:original_vertex_count], faces_t).amin()
        )
    if initial_original_area <= 0.0:
        raise RuntimeError("model initialization did not preserve positive original faces")

    controller = None
    if train_config["lr_schedule"] == "adaptive-plateau":
        controller = AdaptivePlateauController(
            initial_learning_rate=float(train_config["lr"]),
            minimum_learning_rate=float(train_config["min_lr"]),
            initial_minimum_area=initial_original_area,
            window=int(train_config["plateau_window"]),
            patience=int(train_config["plateau_patience"]),
            relative_threshold=float(train_config["plateau_relative_threshold"]),
            factor=float(train_config["plateau_factor"]),
            q_threshold=float(train_config["plateau_q_threshold"]),
            minimum_area_ratio=float(train_config["plateau_minimum_area_ratio"]),
        )

    history: list[dict[str, Any]] = []
    learning_rate_events: list[dict[str, Any]] = []
    check_interval = int(train_config["check_interval"])
    iterations = int(train_config["iters"])
    start_time = time.perf_counter()
    final_diagnostics = None
    final_uv = None

    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        output_uv, diagnostics = model(return_diagnostics=True)
        original_uv = output_uv[:original_vertex_count]
        loss = symmetric_dirichlet_loss(vertices_t, faces_t, original_uv)
        original_areas = signed_double_areas(original_uv, faces_t)
        extended_areas = signed_double_areas(output_uv, model_faces)

        finite = bool(torch.isfinite(output_uv).all() and torch.isfinite(loss))
        positive = bool(torch.all(extended_areas > 0.0))
        if not finite or not positive:
            raise RuntimeError(
                f"hard-validity assertion failed at iteration {iteration}; "
                "training stopped without rollback"
            )

        q_values = diagnostics.q_values.detach()
        q_max = float(q_values.amax())
        q_p95 = float(torch.quantile(q_values, 0.95))
        minimum_original_area = float(original_areas.detach().amin())

        if iteration % check_interval == 0 or iteration == iterations:
            history.append(
                {
                    "iteration": int(iteration),
                    "phase": "adam_no_rollback",
                    "loss": float(loss.detach()),
                    "loss_distortion": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "q_p95": q_p95,
                    "q_max": q_max,
                    "minimum_signed_area": 0.5 * minimum_original_area,
                    "minimum_extended_signed_area": 0.5 * float(extended_areas.detach().amin()),
                    "is_valid": True,
                }
            )

        if iteration == iterations:
            final_uv = output_uv.detach()
            final_diagnostics = diagnostics
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(train_config["gradient_clip"])
        )
        optimizer.step()

        if controller is not None:
            event = controller.observe(
                step=iteration,
                loss=float(loss.detach()),
                q_max=q_max,
                minimum_area=minimum_original_area,
            )
            if event is not None:
                for group in optimizer.param_groups:
                    group["lr"] = event.new_learning_rate
                learning_rate_events.append(
                    {
                        "iteration": int(event.step),
                        "reason": event.reason,
                        "old_learning_rate": event.old_learning_rate,
                        "new_learning_rate": event.new_learning_rate,
                        "relative_improvement": event.relative_improvement,
                        "recent_q_p95": event.recent_q_p95,
                        "recent_area_ratio": event.recent_area_ratio,
                    }
                )

    assert final_uv is not None and final_diagnostics is not None
    with torch.no_grad():
        restored = model(final_uv, inverse=True)
    inverse_error = float(torch.max(torch.abs(restored - model_initial_uv)))
    q_summary = _q_summary(
        final_diagnostics.q_values.detach(),
        final_diagnostics.vertex_ids.detach(),
        scaffold.original_boundary if scaffold is not None else None,
    )
    info = {
        "selected_checkpoint": "final_iteration_no_rollback",
        "selected_iteration": iterations,
        "elapsed_seconds": time.perf_counter() - start_time,
        "rollback_enabled": False,
        "hard_validity_assertions": True,
        "inverse_max_abs_error": inverse_error,
        "learning_rate_events": learning_rate_events,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "topology": topology,
        "model": {
            "class": type(model).__name__,
            "cycles": int(model.cycles),
            "color_count": int(model.color_count),
            "coupling_layer_count": len(model.conditioners),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "conditioner_features": model.feature_set,
        },
        "scaffold": {
            "enabled": scaffold is not None,
            "scale": float(scaffold_config["scale"]) if scaffold is not None else None,
            "outer_boundary_vertex_count": (
                int(scaffold.outer_boundary.numel()) if scaffold is not None else 0
            ),
            "loss_includes_scaffold_faces": False,
        },
        "q": q_summary,
        "objective": {
            "distortion": "3d_face_area_weighted_symmetric_dirichlet",
            "faces_in_objective": "original_mesh_only",
            "jacobian_barrier": False,
            "intersection_penalty": False,
        },
    }
    return MeshPLTrainingResult(
        uv=final_uv[:original_vertex_count].cpu().numpy(),
        model=model,
        history=history,
        info=info,
        final_diagnostics=final_diagnostics,
        model_initial_uv=model_initial_uv.detach(),
        model_final_uv=final_uv,
        model_faces=model_faces,
        scaffold=scaffold,
    )


def _q_summary(q_values: Tensor, vertex_ids: Tensor, original_boundary: Tensor | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "all_updates": _distribution(q_values),
        "interpretation": "conditioning/legality-margin diagnostic, not a distortion metric",
    }
    if original_boundary is None:
        summary["original_interior_updates"] = _distribution(q_values)
        summary["released_source_boundary_updates"] = None
        return summary
    boundary_mask = torch.isin(vertex_ids, original_boundary)
    summary["original_interior_updates"] = _distribution(q_values[~boundary_mask])
    summary["released_source_boundary_updates"] = _distribution(q_values[boundary_mask])
    return summary


def _distribution(values: Tensor) -> dict[str, float | int] | None:
    if values.numel() == 0:
        return None
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.amax()),
    }
