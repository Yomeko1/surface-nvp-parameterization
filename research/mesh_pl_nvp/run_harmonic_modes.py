"""Run the isolated certified global harmonic-mode PL-NVP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from surface_nvp.init_param import generate_initial_uv
from surface_nvp.injectivity.validators import validate_uv, validate_uv_torch
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.losses.distortion import symmetric_dirichlet_loss
from surface_nvp.training.metrics import compute_distortion_metrics
from surface_nvp.training.summary import build_run_summary, save_run_summary
from surface_nvp.visualization.plot_training import save_loss_plot
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot, save_uv_plot
from surface_nvp.visualization.uv_diagnostics import (
    save_area_comparison_heatmap,
    save_distortion_comparison_heatmap,
    save_distortion_heatmap,
    save_flip_heatmap,
    save_intersection_heatmap,
)

from .harmonic_modes import (
    GlobalHarmonicModeFlow,
    build_radial_fourier_harmonic_modes,
    preflight_mode_intervals,
)
from .mesh_coupling import signed_double_areas
from .pipeline_training import validate_disk_topology
from .scaffold import build_outer_scaffold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certified global harmonic-mode mesh PL-NVP ablation"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prim-path", default=None)
    parser.add_argument("--frequencies", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--max-log-scale", type=float, default=0.08)
    parser.add_argument("--max-shift", type=float, default=0.20)
    parser.add_argument("--area-margin-ratio", type=float, default=1.0e-6)
    parser.add_argument("--scaffold-scale", type=float, default=1.1)
    parser.add_argument("--check-interval", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--intersection-batch-size", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    return parser


def _boundary_summary(initial_uv, final_uv, boundary) -> dict[str, float]:
    initial = initial_uv[boundary]
    final = final_uv[boundary]
    displacement = torch.linalg.vector_norm(final - initial, dim=-1)
    initial_center = initial.mean(dim=0)
    initial_radius = torch.linalg.vector_norm(
        initial - initial_center, dim=-1
    ).mean().clamp_min(1.0e-15)
    final_center = final.mean(dim=0)
    final_radii = torch.linalg.vector_norm(final - final_center, dim=-1)
    return {
        "mean_displacement": float(displacement.mean()),
        "maximum_displacement": float(displacement.amax()),
        "mean_displacement_percent_of_initial_radius": float(
            100.0 * displacement.mean() / initial_radius
        ),
        "maximum_displacement_percent_of_initial_radius": float(
            100.0 * displacement.amax() / initial_radius
        ),
        "radial_cv_percent": float(
            100.0
            * final_radii.std(unbiased=False)
            / final_radii.mean().clamp_min(1.0e-15)
        ),
    }


def _finite_number(value: torch.Tensor) -> float | None:
    return float(value.detach()) if bool(torch.isfinite(value.detach())) else None


def save_boundary_comparison(
    path: Path,
    initial_uv: np.ndarray,
    final_uv: np.ndarray,
    boundary: np.ndarray,
) -> None:
    """Make the low-frequency boundary change visible without mesh clutter."""
    import matplotlib.pyplot as plt

    closed = np.concatenate((boundary, boundary[:1]))
    initial = initial_uv[closed]
    final = final_uv[closed]
    figure, axis = plt.subplots(figsize=(7.2, 7.2))
    axis.plot(initial[:, 0], initial[:, 1], "--", color="0.55", linewidth=2.0, label="Tutte")
    axis.plot(final[:, 0], final[:, 1], color="#1565c0", linewidth=2.2, label="Harmonic-mode PL-NVP")
    axis.set_aspect("equal")
    axis.set_title("Source-boundary deformation")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    if args.iters < 0 or args.cycles <= 0 or args.lr <= 0.0:
        raise ValueError("iters must be non-negative; cycles and lr must be positive")
    if args.scaffold_scale <= 1.0:
        raise ValueError("scaffold-scale must be greater than one")
    total_start = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.input, prim_path=args.prim_path)
    topology = validate_disk_topology(mesh.faces, len(mesh.vertices))
    uv0 = generate_initial_uv(
        mesh,
        method="tutte",
        boundary_mode="circle",
        geometry_scale=True,
    ).uv
    initial_injectivity = validate_uv(uv0, mesh.faces)
    if not initial_injectivity["is_valid"]:
        raise ValueError("global harmonic modes require a legal Tutte initialization")
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)

    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float64)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long)
    initial_uv = torch.as_tensor(uv0, dtype=torch.float64)
    scaffold = build_outer_scaffold(
        vertices,
        faces,
        initial_uv,
        scale=args.scaffold_scale,
    )
    names, modes = build_radial_fourier_harmonic_modes(
        scaffold.uv,
        scaffold.faces,
        scaffold.original_boundary,
        scaffold.outer_boundary,
        frequencies=args.frequencies,
    )
    preflight = preflight_mode_intervals(
        scaffold.uv,
        scaffold.faces,
        modes,
        names,
        area_margin_ratio=args.area_margin_ratio,
    )
    preprocessing_seconds = time.perf_counter() - total_start
    print(
        f"initial valid=True SD={initial_distortion['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"V={topology['vertex_count']} F={topology['face_count']}"
    )
    print(
        "preflight minimum_clearance="
        f"{min(float(row['minimum_clearance']) for row in preflight):.8f} "
        f"modes={len(names)}"
    )

    model = GlobalHarmonicModeFlow(
        scaffold.uv,
        scaffold.faces,
        modes,
        names,
        cycles=args.cycles,
        max_log_scale=args.max_log_scale,
        max_shift=args.max_shift,
        area_margin_ratio=args.area_margin_ratio,
    ).to(device=device, dtype=torch.float64)
    original_vertices = vertices.to(device)
    original_faces = faces.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, object]] = []
    training_start = time.perf_counter()
    final_uv = None
    final_diagnostics = None
    for iteration in range(args.iters + 1):
        optimizer.zero_grad(set_to_none=True)
        mapped, diagnostics = model(return_diagnostics=True)
        original_uv = mapped[: len(mesh.vertices)]
        loss = symmetric_dirichlet_loss(
            original_vertices,
            original_faces,
            original_uv,
        )
        extended_areas = signed_double_areas(mapped, model.faces)
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"non-finite loss at iteration {iteration}")
        if not bool(torch.all(extended_areas.detach() > model.area_floor.detach())):
            raise RuntimeError(
                f"hard area assertion failed at iteration {iteration}; no rollback attempted"
            )
        if iteration % args.check_interval == 0 or iteration == args.iters:
            boundary = scaffold.original_boundary.to(device)
            motion = _boundary_summary(model.initial_uv, mapped.detach(), boundary)
            history.append(
                {
                    "iteration": iteration,
                    "phase": "global_harmonic_modes_no_rollback",
                    "loss": float(loss.detach()),
                    "loss_distortion": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "q_max": float(diagnostics.q_values.detach().amax()),
                    "q_p95": float(
                        torch.quantile(diagnostics.q_values.detach(), 0.95)
                    ),
                    "minimum_extended_signed_area": float(
                        0.5 * extended_areas.detach().amin()
                    ),
                    "source_boundary_displacement_percent": motion[
                        "mean_displacement_percent_of_initial_radius"
                    ],
                    "source_boundary_radial_cv_percent": motion[
                        "radial_cv_percent"
                    ],
                    "is_valid": True,
                }
            )
        if iteration == args.iters:
            final_uv = mapped.detach()
            final_diagnostics = diagnostics
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
    training_seconds = time.perf_counter() - training_start
    assert final_uv is not None and final_diagnostics is not None

    with torch.no_grad():
        restored = model(final_uv, inverse=True)
    inverse_error = float(torch.max(torch.abs(restored - model.initial_uv)))
    final_original_uv = final_uv[: len(mesh.vertices)].cpu().numpy()
    final_injectivity = validate_uv(final_original_uv, mesh.faces)
    extended_injectivity = validate_uv_torch(
        final_uv,
        model.faces,
        intersection_batch_size=args.intersection_batch_size,
    )
    if not final_injectivity["is_valid"] or not extended_injectivity["is_valid"]:
        raise RuntimeError("final injectivity audit failed; no result was saved")
    final_distortion = compute_distortion_metrics(
        mesh.vertices,
        mesh.faces,
        final_original_uv,
    )
    boundary_motion = _boundary_summary(
        model.initial_uv,
        final_uv,
        scaffold.original_boundary.to(device),
    )
    coefficient_rows = []
    for index, name in enumerate(final_diagnostics.names):
        coefficient_rows.append(
            {
                "layer": name,
                "before": float(final_diagnostics.coefficients_before[index].detach()),
                "after": float(final_diagnostics.coefficients_after[index].detach()),
                "lower": _finite_number(final_diagnostics.lower_bounds[index]),
                "upper": _finite_number(final_diagnostics.upper_bounds[index]),
                "q": float(final_diagnostics.q_values[index].detach()),
            }
        )
    info = {
        "selected_checkpoint": "final_iteration_no_rollback",
        "selected_iteration": args.iters,
        "rollback_enabled": False,
        "hard_validity_assertions": True,
        "inverse_max_abs_error": inverse_error,
        "elapsed_seconds": training_seconds,
        "topology": topology,
        "model": {
            "class": type(model).__name__,
            "frequencies": args.frequencies,
            "base_modes": list(names),
            "cycles": args.cycles,
            "coupling_layer_count": len(names) * args.cycles,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "outer_scaffold_boundary_fixed": True,
        },
        "objective": {
            "distortion": "3d_face_area_weighted_symmetric_dirichlet",
            "faces_in_objective": "original_mesh_only",
            "jacobian_barrier": False,
            "intersection_penalty": False,
        },
        "preflight": preflight,
        "final_layers": coefficient_rows,
        "boundary_motion": boundary_motion,
        "final_extended_injectivity": extended_injectivity,
    }

    artifact_start = time.perf_counter()
    save_mesh(output, mesh, uv=final_original_uv)
    save_uv_plot(output.with_suffix(".initial.uv.png"), uv0, mesh.faces)
    save_uv_plot(output.with_suffix(".uv.png"), final_original_uv, mesh.faces)
    save_uv_comparison_plot(
        output.with_suffix(".compare.png"),
        uv0,
        final_original_uv,
        mesh.faces,
        initial_title=(
            "Initial UV\n"
            f"SD={initial_distortion['symmetric_dirichlet_area_weighted_mean']:.4g}"
        ),
        final_title=(
            "Global harmonic-mode PL-NVP (no rollback)\n"
            f"SD={final_distortion['symmetric_dirichlet_area_weighted_mean']:.4g}"
        ),
    )
    save_boundary_comparison(
        output.with_suffix(".boundary_compare.png"),
        uv0,
        final_original_uv,
        scaffold.original_boundary.cpu().numpy(),
    )
    save_flip_heatmap(
        output.with_suffix(".flip_heatmap.png"),
        final_original_uv,
        mesh.faces,
        title="Global Harmonic-Mode PL-NVP Flip Heatmap",
    )
    save_area_comparison_heatmap(
        output.with_suffix(".area_compare.png"), uv0, final_original_uv, mesh.faces
    )
    save_distortion_heatmap(
        output.with_suffix(".distortion.png"),
        mesh.vertices,
        mesh.faces,
        final_original_uv,
        title="Global Harmonic-Mode Symmetric Dirichlet",
    )
    save_distortion_comparison_heatmap(
        output.with_suffix(".distortion_compare.png"),
        mesh.vertices,
        mesh.faces,
        uv0,
        final_original_uv,
    )
    save_intersection_heatmap(
        output.with_suffix(".intersection_heatmap.png"),
        final_original_uv,
        mesh.faces,
        final_injectivity["intersections"],
    )
    save_loss_plot(output.with_suffix(".loss.png"), history)
    torch.save(model.state_dict(), output.with_suffix(".model.pt"))
    runtime = {
        "preprocessing_seconds": preprocessing_seconds,
        "optimization_loop_seconds": training_seconds,
        "artifact_seconds": time.perf_counter() - artifact_start,
        "total_seconds": time.perf_counter() - total_start,
    }
    info["runtime"] = runtime
    config = vars(args)
    with output.with_suffix(".config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    with output.with_suffix(".runtime.json").open("w", encoding="utf-8") as stream:
        json.dump(runtime, stream, indent=2)
    payload = {
        "initial": {"injectivity": initial_injectivity, "distortion": initial_distortion},
        "final": {"injectivity": final_injectivity, "distortion": final_distortion},
        "training": info,
        "history": history,
    }
    with output.with_suffix(".metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    save_run_summary(
        output,
        build_run_summary("global_harmonic_mode_pl_nvp", args.iters, payload),
    )
    print(
        f"final valid=True SD={final_distortion['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"q_max={float(final_diagnostics.q_values.detach().amax()):.6f} "
        f"boundary_move={boundary_motion['mean_displacement_percent_of_initial_radius']:.4f}% "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
