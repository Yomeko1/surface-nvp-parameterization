"""Train the standalone mesh coupling flow on a small real patch of Balls."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .adaptive_lr import AdaptivePlateauController
from .mesh_coupling import count_proper_edge_intersections, signed_double_areas
from .batched_coupling import BatchedMeshCouplingFlow
from .scaffold import build_outer_scaffold
from .trainable_coupling import MeshCouplingFlow


DTYPE = torch.float64


def delayed_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    decay_start: int,
    minimum_ratio: float,
) -> float:
    """Hold the base LR, then smoothly decay to a requested fraction."""
    if step <= decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (
        1.0 + math.cos(math.pi * progress)
    )


def load_obj_with_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal standalone OBJ reader for one UV value per geometric vertex."""
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_texcoords: list[list[int | None]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "vt":
                texcoords.append([float(value) for value in fields[1:3]])
            elif fields[0] == "f":
                parsed: list[tuple[int, int | None]] = []
                for token in fields[1:]:
                    parts = token.split("/")
                    vertex = int(parts[0]) - 1
                    texcoord = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                    parsed.append((vertex, texcoord))
                for index in range(1, len(parsed) - 1):
                    triangle = (parsed[0], parsed[index], parsed[index + 1])
                    faces.append([entry[0] for entry in triangle])
                    face_texcoords.append([entry[1] for entry in triangle])
    if not texcoords:
        raise ValueError(f"OBJ does not contain UV coordinates: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    texcoord_array = np.asarray(texcoords, dtype=np.float64)
    uv_sum = np.zeros((len(vertices), 2), dtype=np.float64)
    uv_count = np.zeros(len(vertices), dtype=np.int64)
    for triangle, triangle_texcoords in zip(faces, face_texcoords, strict=True):
        for vertex, texcoord in zip(triangle, triangle_texcoords, strict=True):
            if texcoord is not None:
                uv_sum[vertex] += texcoord_array[texcoord]
                uv_count[vertex] += 1
    if np.any(uv_count == 0):
        raise ValueError("some OBJ vertices have no UV coordinate")
    uv = uv_sum / uv_count[:, None]
    return vertex_array, face_array, uv


def _reference_data(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    triangles = vertices[faces]
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    length1 = torch.linalg.vector_norm(e1, dim=-1).clamp_min(1.0e-12)
    x2 = torch.sum(e1 * e2, dim=-1) / length1
    y2 = torch.sqrt((torch.sum(e2.square(), dim=-1) - x2.square()).clamp_min(1.0e-12))
    reference = torch.zeros((faces.shape[0], 2, 2), dtype=vertices.dtype, device=vertices.device)
    reference[:, 0, 0] = length1
    reference[:, 0, 1] = x2
    reference[:, 1, 1] = y2
    areas = 0.5 * torch.linalg.vector_norm(torch.linalg.cross(e1, e2, dim=-1), dim=-1)
    return torch.linalg.inv(reference), areas


def jacobian_singular_values(
    uv: torch.Tensor, faces: torch.Tensor, inverse_reference: torch.Tensor
) -> torch.Tensor:
    triangles = uv[faces]
    uv_edges = torch.stack(
        (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), dim=-1
    )
    return torch.linalg.svdvals(uv_edges @ inverse_reference)


def symmetric_dirichlet(
    uv: torch.Tensor,
    faces: torch.Tensor,
    inverse_reference: torch.Tensor,
    areas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    singular_values = jacobian_singular_values(uv, faces, inverse_reference)
    inverse_values = (singular_values + 1.0e-4).reciprocal()
    per_face = (singular_values.square() + inverse_values.square()).sum(dim=-1)
    loss = torch.sum(per_face * areas) / areas.sum().clamp_min(1.0e-12)
    return loss, per_face, singular_values


def _face_adjacency(faces: np.ndarray) -> tuple[list[set[int]], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for offset in range(3):
            edge = tuple(sorted((int(face[offset]), int(face[(offset + 1) % 3]))))
            edge_faces[edge].append(face_index)
    adjacency = [set() for _ in range(len(faces))]
    boundary_faces: set[int] = set()
    for incident in edge_faces.values():
        if len(incident) == 1:
            boundary_faces.add(incident[0])
        elif len(incident) == 2:
            a, b = incident
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency, sorted(boundary_faces)


def _dual_distances(adjacency: list[set[int]], sources: list[int]) -> list[int]:
    distance = [-1] * len(adjacency)
    queue: deque[int] = deque()
    for source in sources:
        distance[source] = 0
        queue.append(source)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def _patch_topology(faces: np.ndarray, vertex_count: int) -> tuple[int, list[list[int]]]:
    edges = Counter(
        tuple(sorted((int(face[offset]), int(face[(offset + 1) % 3]))))
        for face in faces
        for offset in range(3)
    )
    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    euler = vertex_count - len(edges) + len(faces)
    return euler, components


def extract_distorted_disk_patch(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    *,
    rings: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    vertices_t = torch.tensor(vertices, dtype=DTYPE)
    faces_t = torch.tensor(faces, dtype=torch.long)
    uv_t = torch.tensor(uv, dtype=DTYPE)
    inverse_reference, areas = _reference_data(vertices_t, faces_t)
    _, distortion, _ = symmetric_dirichlet(uv_t, faces_t, inverse_reference, areas)
    adjacency, boundary_faces = _face_adjacency(faces)
    distance_to_boundary = _dual_distances(adjacency, boundary_faces)
    order = torch.argsort(distortion, descending=True).detach().cpu().tolist()

    for seed_face in order:
        if distance_to_boundary[seed_face] < rings + 1:
            continue
        patch_faces: set[int] = {seed_face}
        frontier = {seed_face}
        for _ in range(rings):
            frontier = {neighbor for face in frontier for neighbor in adjacency[face]} - patch_faces
            patch_faces.update(frontier)
        selected_faces = faces[sorted(patch_faces)]
        selected_vertices = np.unique(selected_faces.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[selected_vertices] = np.arange(len(selected_vertices))
        local_faces = remap[selected_faces]
        euler, boundary_components = _patch_topology(local_faces, len(selected_vertices))
        if euler == 1 and len(boundary_components) == 1:
            metadata = {
                "seed_face": int(seed_face),
                "rings": int(rings),
                "source_vertex_count": int(len(vertices)),
                "source_face_count": int(len(faces)),
                "patch_vertex_count": int(len(selected_vertices)),
                "patch_face_count": int(len(local_faces)),
                "patch_boundary_vertex_count": int(len(boundary_components[0])),
            }
            return vertices[selected_vertices], local_faces, uv[selected_vertices], metadata
    raise RuntimeError("could not extract a disk patch away from the source boundary")


def distortion_metrics(
    uv: torch.Tensor,
    faces: torch.Tensor,
    inverse_reference: torch.Tensor,
    areas: torch.Tensor,
) -> dict[str, float | int]:
    loss, per_face, singular_values = symmetric_dirichlet(uv, faces, inverse_reference, areas)
    condition = singular_values[:, 0] / singular_values[:, 1].clamp_min(1.0e-12)
    double_areas = signed_double_areas(uv, faces)
    return {
        "symmetric_dirichlet_area_weighted_mean": float(loss.detach()),
        "symmetric_dirichlet_mean": float(per_face.detach().mean()),
        "symmetric_dirichlet_p95": float(torch.quantile(per_face.detach(), 0.95)),
        "symmetric_dirichlet_max": float(per_face.detach().max()),
        "condition_number_mean": float(condition.detach().mean()),
        "condition_number_p95": float(torch.quantile(condition.detach(), 0.95)),
        "minimum_double_area": float(double_areas.detach().min()),
        "flipped_faces": int(torch.count_nonzero(double_areas.detach() <= 0.0)),
        "proper_edge_intersections": count_proper_edge_intersections(uv.detach(), faces),
    }


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Standalone mesh_pl_nvp Balls patch experiment\n")
        for vertex in vertices:
            stream.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for texcoord in uv:
            stream.write(f"vt {texcoord[0]:.17g} {texcoord[1]:.17g}\n")
        for face in faces + 1:
            stream.write(
                f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n"
            )


def plot_result(
    initial_uv: torch.Tensor,
    final_uv: torch.Tensor,
    faces: torch.Tensor,
    history: list[dict[str, float]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    for axis, uv, title in zip(
        axes[:2], (initial_uv, final_uv), ("initial UV", "optimized PL-NVP UV"), strict=True
    ):
        points = uv.detach().cpu().numpy()
        for face in faces.detach().cpu().numpy():
            loop = np.append(face, face[0])
            axis.plot(points[loop, 0], points[loop, 1], color="#333333", linewidth=0.55)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.axis("off")
    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]
    axes[2].plot(steps, losses, color="#4c78a8")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("area-weighted SD")
    axes[2].set_title("training trajectory")
    axes[2].grid(alpha=0.25)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def q_hotspot_report(
    diagnostics,
    model: MeshCouplingFlow,
    initial_uv: torch.Tensor,
    final_uv: torch.Tensor,
    faces: torch.Tensor,
    per_face_sd: torch.Tensor,
    *,
    top_k: int = 10,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    """Locate the distinct vertices with the largest intermediate radial fraction q."""
    q_values = diagnostics.q_values.detach()
    vertex_ids = diagnostics.vertex_ids.detach()
    layer_ids = diagnostics.layer_ids.detach()
    vertex_q = torch.zeros(
        initial_uv.shape[0], dtype=q_values.dtype, device=q_values.device
    )
    vertex_q.scatter_reduce_(0, vertex_ids, q_values, reduce="amax", include_self=True)

    records: list[dict[str, object]] = []
    seen: set[int] = set()
    for diagnostic_index in torch.argsort(q_values, descending=True).cpu().tolist():
        vertex = int(vertex_ids[diagnostic_index])
        if vertex in seen:
            continue
        seen.add(vertex)
        layer = int(layer_ids[diagnostic_index])
        neighbors = sorted(model._adjacency[vertex])
        incident = torch.nonzero(
            torch.any(faces == vertex, dim=1), as_tuple=False
        ).flatten()
        displacement = torch.linalg.vector_norm(final_uv[vertex] - initial_uv[vertex])
        records.append(
            {
                "rank": len(records) + 1,
                "vertex_id": vertex,
                "q": float(q_values[diagnostic_index]),
                "halfplane_slack": float(
                    diagnostics.halfplane_slacks[diagnostic_index].detach()
                ),
                "layer_id": layer,
                "cycle": layer // model.color_count,
                "color": layer % model.color_count,
                "degree": len(neighbors),
                "neighbor_vertex_ids": neighbors,
                "incident_face_ids": incident.cpu().tolist(),
                "incident_sd_max": float(per_face_sd[incident].max()),
                "initial_uv": initial_uv[vertex].detach().cpu().tolist(),
                "final_uv": final_uv[vertex].detach().cpu().tolist(),
                "displacement": float(displacement),
            }
        )
        if len(records) >= top_k:
            break
    return records, vertex_q


def plot_q_hotspots(
    final_uv: torch.Tensor,
    faces: torch.Tensor,
    vertex_q: torch.Tensor,
    hotspots: list[dict[str, object]],
    output: Path,
) -> None:
    points = final_uv.detach().cpu().numpy()
    q = vertex_q.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    for face in faces.detach().cpu().numpy():
        loop = np.append(face, face[0])
        axis.plot(points[loop, 0], points[loop, 1], color="#b7b7b7", linewidth=0.4, zorder=1)
    scatter = axis.scatter(
        points[:, 0], points[:, 1], c=q, cmap="magma", s=11.0, vmin=0.0, vmax=1.0, zorder=2
    )
    for record in hotspots[:5]:
        vertex = int(record["vertex_id"])
        axis.annotate(
            str(vertex), points[vertex], xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    figure.colorbar(scatter, ax=axis, label="maximum q over coupling layers")
    axis.set_title("radial-fraction hotspots (top vertex IDs labelled)")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/output/v2.4/Balls/Balls/initial/Balls_initial.obj"),
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts" / "balls_patch")
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument(
        "--full-mesh",
        action="store_true",
        help="train on the complete Balls disk instead of an extracted local patch",
    )
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--conditioner-features",
        choices=("basic", "local-geometry"),
        default="basic",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "delayed-cosine", "adaptive-plateau"),
        default="constant",
    )
    parser.add_argument("--lr-decay-start", type=int, default=300)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--plateau-window", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-relative-threshold", type=float, default=8.0e-3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-q-threshold", type=float, default=0.97)
    parser.add_argument("--plateau-minimum-area-ratio", type=float, default=0.25)
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="free the source boundary by fixing a convex outer scaffold ring",
    )
    parser.add_argument("--scaffold-scale", type=float, default=1.1)
    parser.add_argument(
        "--initial-model-state",
        type=Path,
        help="optional model_state.pt used to initialize a refinement run",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--reference-unbatched",
        action="store_true",
        help="use the slow per-vertex reference implementation",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source_vertices, source_faces, source_uv = load_obj_with_uv(args.input)
    if args.full_mesh:
        euler, boundary_components = _patch_topology(source_faces, len(source_vertices))
        if euler != 1 or len(boundary_components) != 1:
            raise ValueError("--full-mesh requires a single-boundary topological disk")
        patch_vertices, patch_faces, patch_uv = source_vertices, source_faces, source_uv
        patch_metadata = {
            "seed_face": -1,
            "rings": -1,
            "source_vertex_count": int(len(source_vertices)),
            "source_face_count": int(len(source_faces)),
            "patch_vertex_count": int(len(source_vertices)),
            "patch_face_count": int(len(source_faces)),
            "patch_boundary_vertex_count": int(len(boundary_components[0])),
        }
    else:
        patch_vertices, patch_faces, patch_uv, patch_metadata = extract_distorted_disk_patch(
            source_vertices, source_faces, source_uv, rings=args.rings
        )

    vertices = torch.tensor(patch_vertices, dtype=DTYPE)
    faces = torch.tensor(patch_faces, dtype=torch.long)
    initial_uv = torch.tensor(patch_uv, dtype=DTYPE)
    inverse_reference, areas = _reference_data(vertices, faces)
    model_vertices = vertices
    model_faces = faces
    model_initial_uv = initial_uv
    scaffold = None
    if args.scaffold:
        scaffold = build_outer_scaffold(
            vertices, faces, initial_uv, scale=args.scaffold_scale
        )
        model_vertices = scaffold.vertices_3d
        model_faces = scaffold.faces
        model_initial_uv = scaffold.uv
        patch_metadata["scaffold_vertex_count"] = int(
            model_vertices.shape[0] - vertices.shape[0]
        )
        patch_metadata["scaffold_face_count"] = int(
            model_faces.shape[0] - faces.shape[0]
        )
    model_type = MeshCouplingFlow if args.reference_unbatched else BatchedMeshCouplingFlow
    model = model_type(
        model_vertices,
        model_faces,
        model_initial_uv,
        cycles=args.cycles,
        hidden_dim=args.hidden_dim,
        feature_set=args.conditioner_features,
    ).to(dtype=DTYPE)
    if args.initial_model_state is not None:
        state = torch.load(args.initial_model_state, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("--minimum-learning-rate must lie in (0, --learning-rate]")
    if args.lr_schedule == "delayed-cosine" and not 0 <= args.lr_decay_start <= args.iterations:
        raise ValueError("--lr-decay-start must lie in [0, --iterations]")
    scheduler = None
    if args.lr_schedule == "delayed-cosine":
        minimum_ratio = args.minimum_learning_rate / args.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: delayed_cosine_multiplier(
                step,
                total_steps=args.iterations,
                decay_start=args.lr_decay_start,
                minimum_ratio=minimum_ratio,
            ),
        )

    initial_metrics = distortion_metrics(initial_uv, faces, inverse_reference, areas)
    with torch.no_grad():
        optimization_start_uv = model()
    optimization_start_metrics = distortion_metrics(
        optimization_start_uv[: vertices.shape[0]], faces, inverse_reference, areas
    )
    adaptive_controller = None
    if args.lr_schedule == "adaptive-plateau":
        adaptive_controller = AdaptivePlateauController(
            initial_learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            initial_minimum_area=optimization_start_metrics["minimum_double_area"],
            window=args.plateau_window,
            patience=args.plateau_patience,
            relative_threshold=args.plateau_relative_threshold,
            factor=args.plateau_factor,
            q_threshold=args.plateau_q_threshold,
            minimum_area_ratio=args.plateau_minimum_area_ratio,
        )
    history: list[dict[str, float]] = []
    learning_rate_events: list[dict[str, float | str]] = []
    start = time.perf_counter()
    for step in range(args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        output_uv, diagnostics = model(return_diagnostics=True)
        original_output_uv = output_uv[: vertices.shape[0]]
        loss, _, _ = symmetric_dirichlet(
            original_output_uv, faces, inverse_reference, areas
        )
        areas2 = signed_double_areas(original_output_uv, faces)
        if step < args.iterations:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if adaptive_controller is not None:
                event = adaptive_controller.observe(
                    step=step,
                    loss=float(loss.detach()),
                    q_max=float(diagnostics.q_values.detach().max()),
                    minimum_area=float(areas2.detach().min()),
                )
                if event is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = event.new_learning_rate
                    event_record = {
                        "step": float(event.step),
                        "old_learning_rate": event.old_learning_rate,
                        "new_learning_rate": event.new_learning_rate,
                        "reason": event.reason,
                        "relative_improvement": event.relative_improvement,
                        "recent_q_p95": event.recent_q_p95,
                        "recent_area_ratio": event.recent_area_ratio,
                    }
                    learning_rate_events.append(event_record)
                    print(
                        f"lr_event step={event.step:04d} reason={event.reason} "
                        f"relative={event.relative_improvement:.3e} "
                        f"lr={event.old_learning_rate:.3e}->{event.new_learning_rate:.3e}"
                    )
        if step % 10 == 0 or step == args.iterations:
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "q_max": float(diagnostics.q_values.detach().max()),
                "q_p95": float(torch.quantile(diagnostics.q_values.detach(), 0.95)),
                "minimum_double_area": float(areas2.detach().min()),
                "minimum_halfplane_slack": float(diagnostics.halfplane_slacks.detach().min()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            print(
                f"step={step:04d} loss={record['loss']:.8f} "
                f"q95={record['q_p95']:.6f} qmax={record['q_max']:.6f} "
                f"min_area2={record['minimum_double_area']:.3e}"
            )

    elapsed = time.perf_counter() - start
    with torch.no_grad():
        final_uv, final_diagnostics = model(return_diagnostics=True)
        restored_uv = model(final_uv, inverse=True)
    final_original_uv = final_uv[: vertices.shape[0]]
    final_metrics = distortion_metrics(
        final_original_uv, faces, inverse_reference, areas
    )
    _, final_per_face_sd, _ = symmetric_dirichlet(
        final_original_uv, faces, inverse_reference, areas
    )
    hotspots, vertex_q = q_hotspot_report(
        final_diagnostics,
        model,
        model_initial_uv,
        final_uv,
        faces,
        final_per_face_sd,
    )
    inverse_error = float(torch.max(torch.abs(restored_uv - model_initial_uv)))
    scaffold_summary = None
    if scaffold is not None:
        boundary_displacement = torch.linalg.vector_norm(
            final_uv[scaffold.original_boundary]
            - model_initial_uv[scaffold.original_boundary],
            dim=-1,
        )
        extended_areas = signed_double_areas(final_uv, model_faces)
        scaffold_summary = {
            "scale": args.scaffold_scale,
            "outer_boundary_vertices": int(scaffold.outer_boundary.shape[0]),
            "source_boundary_displacement_mean": float(boundary_displacement.mean()),
            "source_boundary_displacement_max": float(boundary_displacement.max()),
            "extended_minimum_double_area": float(extended_areas.min()),
            "extended_flipped_faces": int(torch.count_nonzero(extended_areas <= 0.0)),
            "extended_proper_edge_intersections": count_proper_edge_intersections(
                final_uv, model_faces
            ),
        }
    summary = {
        "experiment": "standalone_trainable_mesh_coupling",
        "source": str(args.input),
        "patch": patch_metadata,
        "config": {
            "cycles": args.cycles,
            "hidden_dim": args.hidden_dim,
            "conditioner_features": args.conditioner_features,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "lr_schedule": args.lr_schedule,
            "lr_decay_start": args.lr_decay_start,
            "minimum_learning_rate": args.minimum_learning_rate,
            "plateau_window": args.plateau_window,
            "plateau_patience": args.plateau_patience,
            "plateau_relative_threshold": args.plateau_relative_threshold,
            "plateau_factor": args.plateau_factor,
            "plateau_q_threshold": args.plateau_q_threshold,
            "plateau_minimum_area_ratio": args.plateau_minimum_area_ratio,
            "initial_model_state": (
                str(args.initial_model_state) if args.initial_model_state is not None else None
            ),
            "seed": args.seed,
            "dtype": str(DTYPE),
            "rollback": False,
            "loss": "3d-face-area-weighted symmetric Dirichlet only",
            "boundary": (
                "free source boundary with fixed outer scaffold"
                if args.scaffold
                else "fixed patch boundary"
            ),
            "scaffold": args.scaffold,
            "scaffold_scale": args.scaffold_scale,
            "implementation": "reference_unbatched" if args.reference_unbatched else "same_color_batched",
        },
        "initial": initial_metrics,
        "optimization_start": optimization_start_metrics,
        "final": final_metrics,
        "final_q_max": float(final_diagnostics.q_values.max()),
        "final_q_p95": float(torch.quantile(final_diagnostics.q_values, 0.95)),
        "final_minimum_halfplane_slack": float(final_diagnostics.halfplane_slacks.min()),
        "q_hotspots": hotspots,
        "scaffold": scaffold_summary,
        "inverse_linf": inverse_error,
        "elapsed_seconds": elapsed,
        "history": history,
        "learning_rate_events": learning_rate_events,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output / "model_state.pt")
    save_obj(args.output / "initial_patch.obj", patch_vertices, patch_faces, patch_uv)
    save_obj(
        args.output / "optimized_patch.obj",
        patch_vertices,
        patch_faces,
        final_original_uv.cpu().numpy(),
    )
    if scaffold is not None:
        save_obj(
            args.output / "scaffold_initial.obj",
            model_vertices.cpu().numpy(),
            model_faces.cpu().numpy(),
            model_initial_uv.cpu().numpy(),
        )
        save_obj(
            args.output / "scaffold_optimized.obj",
            model_vertices.cpu().numpy(),
            model_faces.cpu().numpy(),
            final_uv.cpu().numpy(),
        )
    plot_result(initial_uv, final_original_uv, faces, history, args.output / "result.png")
    plot_q_hotspots(
        final_uv,
        model_faces if scaffold is not None else faces,
        vertex_q,
        hotspots,
        args.output / "q_hotspots.png",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
