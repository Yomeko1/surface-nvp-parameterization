"""Run quantitative and visual validation for the standalone psi_K prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .mesh_coupling import coupling_cycle, diagnose_embedding, make_grid_triangulation
from .radial_polytope import (
    analytic_center,
    from_polytope,
    halfplane_polygon_vertices,
    halfplanes_from_ccw_polygon,
    minimum_slack,
    to_polytope,
)


DTYPE = torch.float64


def random_convex_polygon(rng: np.random.Generator, count: int = 32) -> torch.Tensor:
    points = rng.normal(size=(count, 2))
    points[:, 0] *= rng.uniform(0.2, 3.0)
    points[:, 1] *= rng.uniform(0.2, 3.0)
    hull = ConvexHull(points)
    return torch.tensor(points[hull.vertices], dtype=DTYPE)


def validate_random_polygons(seed: int = 20260830) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    max_round_trip = 0.0
    minimum_membership_slack = float("inf")
    minimum_jacobian_determinant = float("inf")
    polygon_count = 24
    samples_per_polygon = 512

    for _ in range(polygon_count):
        polygon = random_convex_polygon(rng)
        A, b = halfplanes_from_ccw_polygon(polygon)
        center = analytic_center(A, b)
        latent = center + 5.0 * torch.randn(
            (samples_per_polygon, 2), generator=generator, dtype=DTYPE
        )
        mapped = to_polytope(latent, A, b, center)
        restored = from_polytope(mapped, A, b, center)
        max_round_trip = max(max_round_trip, float(torch.max(torch.abs(restored - latent))))
        minimum_membership_slack = min(
            minimum_membership_slack, float(minimum_slack(mapped, A, b).min())
        )

        for sample in latent[:8]:
            sample = sample.detach().requires_grad_(True)
            jacobian = torch.autograd.functional.jacobian(
                lambda value: to_polytope(value, A, b, center), sample
            )
            minimum_jacobian_determinant = min(
                minimum_jacobian_determinant, float(torch.linalg.det(jacobian))
            )

    return {
        "polygon_count": polygon_count,
        "samples_per_polygon": samples_per_polygon,
        "maximum_round_trip_linf": max_round_trip,
        "minimum_halfplane_slack": minimum_membership_slack,
        "minimum_psi_inverse_jacobian_determinant": minimum_jacobian_determinant,
    }


def validate_boundary_stress() -> list[dict[str, float]]:
    polygon = torch.tensor(
        [[-2.2, -0.8], [1.7, -1.2], [2.3, 0.4], [0.5, 1.8], [-1.6, 1.2]],
        dtype=DTYPE,
    )
    A, b = halfplanes_from_ccw_polygon(polygon)
    center = analytic_center(A, b)
    direction = torch.tensor([0.83, 0.41], dtype=DTYPE)
    direction = direction / torch.linalg.vector_norm(direction)
    results: list[dict[str, float]] = []
    for radius in (1.0, 10.0, 100.0, 1_000.0, 10_000.0):
        latent = center + radius * direction
        mapped = to_polytope(latent, A, b, center)
        restored = from_polytope(mapped, A, b, center)
        absolute_error = float(torch.max(torch.abs(restored - latent)))
        results.append(
            {
                "latent_radius": radius,
                "round_trip_linf": absolute_error,
                "relative_round_trip_linf": absolute_error / radius,
                "minimum_halfplane_slack": float(minimum_slack(mapped, A, b)),
            }
        )
    return results


def validate_mesh_cycles() -> tuple[list[dict[str, float | int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    initial, faces = make_grid_triangulation(7, 7)
    records: list[dict[str, float | int]] = []
    selected = initial
    for cycle_count in range(1, 5):
        deformed = initial
        for _ in range(cycle_count):
            deformed = coupling_cycle(deformed, faces)
        diagnostics = diagnose_embedding(deformed, faces)

        restored = deformed
        for _ in range(cycle_count):
            restored = coupling_cycle(restored, faces, inverse=True)
        records.append(
            {
                "cycles": cycle_count,
                "minimum_double_area": diagnostics.minimum_double_area,
                "flipped_faces": diagnostics.flipped_faces,
                "proper_edge_intersections": diagnostics.proper_edge_intersections,
                "inverse_linf": float(torch.max(torch.abs(restored - initial))),
            }
        )
        if cycle_count == 3:
            selected = deformed
    return records, initial, selected, faces


def _plot_mapping(output: Path) -> None:
    polygon = torch.tensor(
        [[-2.0, -0.7], [1.5, -1.1], [2.2, 0.5], [0.4, 1.8], [-1.7, 1.0]],
        dtype=DTYPE,
    )
    A, b = halfplanes_from_ccw_polygon(polygon)
    center = analytic_center(A, b)
    polygon_vertices = halfplane_polygon_vertices(A, b).detach().numpy()
    hull = ConvexHull(polygon_vertices)
    ordered = polygon_vertices[hull.vertices]

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].set_title("latent grid in $\\mathbb{R}^2$")
    axes[1].set_title("mapped grid inside $K$")
    grid = np.linspace(-4.0, 4.0, 17)
    line = np.linspace(-4.0, 4.0, 401)
    center_np = center.detach().numpy()
    for fixed in grid:
        horizontal = torch.tensor(
            np.column_stack((line, np.full_like(line, fixed))) + center_np, dtype=DTYPE
        )
        vertical = torch.tensor(
            np.column_stack((np.full_like(line, fixed), line)) + center_np, dtype=DTYPE
        )
        axes[0].plot(horizontal[:, 0], horizontal[:, 1], color="#4c78a8", linewidth=0.45)
        axes[0].plot(vertical[:, 0], vertical[:, 1], color="#f58518", linewidth=0.45)
        mapped_h = to_polytope(horizontal, A, b, center).detach().numpy()
        mapped_v = to_polytope(vertical, A, b, center).detach().numpy()
        axes[1].plot(mapped_h[:, 0], mapped_h[:, 1], color="#4c78a8", linewidth=0.45)
        axes[1].plot(mapped_v[:, 0], mapped_v[:, 1], color="#f58518", linewidth=0.45)
    closed = np.vstack((ordered, ordered[0]))
    axes[1].plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.4)
    axes[1].scatter([center_np[0]], [center_np[1]], s=18, color="black")
    for axis in axes:
        axis.set_aspect("equal")
        axis.grid(False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_mesh(initial: torch.Tensor, deformed: torch.Tensor, faces: torch.Tensor, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    for axis, vertices, title in zip(
        axes, (initial, deformed), ("initial grid", "after 3 coupling cycles"), strict=True
    ):
        array = vertices.detach().numpy()
        for face in faces.detach().numpy():
            loop = np.append(face, face[0])
            axis.plot(array[loop, 0], array[loop, 1], color="#333333", linewidth=0.55)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.axis("off")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="directory for JSON and PNG validation artifacts",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    random_summary = validate_random_polygons()
    boundary_stress = validate_boundary_stress()
    mesh_cycles, initial, deformed, faces = validate_mesh_cycles()
    summary = {
        "dtype": str(DTYPE),
        "random_polygons": random_summary,
        "boundary_stress": boundary_stress,
        "mesh_coupling": mesh_cycles,
    }
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_mapping(args.output / "radial_mapping.png")
    _plot_mesh(initial, deformed, faces, args.output / "mesh_coupling.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
