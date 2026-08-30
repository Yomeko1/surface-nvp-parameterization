"""Run isolated mesh PL-NVP with v2.4-style input, output, and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from surface_nvp.init_param import generate_initial_uv, resolve_initial_uv
from surface_nvp.injectivity.validators import validate_uv, validate_uv_torch
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.losses.distortion import symmetric_dirichlet_per_face
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

from .pipeline_config import apply_cli_overrides, load_pipeline_config
from .pipeline_training import train_mesh_pl_nvp, validate_disk_topology
from .run_balls_patch import plot_q_hotspots, q_hotspot_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated mesh-aligned PL-NVP runner with v2.4-compatible I/O"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-uv", default=None)
    parser.add_argument("--initial-model-state", default=None)
    parser.add_argument("--boundary", choices=["circle", "square"], default=None)
    parser.add_argument("--geometry-scale", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--conditioner-features", choices=["basic", "local-geometry"], default=None
    )
    parser.add_argument("--max-log-scale", type=float, default=None)
    parser.add_argument("--max-shift-fraction", type=float, default=None)
    parser.add_argument("--center-iterations", type=int, default=None)

    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check-interval", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument(
        "--lr-schedule", choices=["constant", "adaptive-plateau"], default=None
    )
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=None)
    parser.add_argument("--plateau-patience", type=int, default=None)
    parser.add_argument("--plateau-relative-threshold", type=float, default=None)
    parser.add_argument("--plateau-factor", type=float, default=None)
    parser.add_argument("--plateau-q-threshold", type=float, default=None)
    parser.add_argument("--plateau-minimum-area-ratio", type=float, default=None)
    parser.add_argument("--intersection-batch-size", type=int, default=None)

    parser.add_argument("--scaffold", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scaffold-scale", type=float, default=None)
    parser.add_argument("--prim-path", default=None)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    total_start = time.perf_counter()
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_pipeline_config(args.config), args)
    init_config = config["init"]
    train_config = config["train"]
    io_config = config["io"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.input, prim_path=io_config["prim_path"])
    topology = validate_disk_topology(mesh.faces, len(mesh.vertices))
    if init_config["initial_uv"] is None:
        uv0 = generate_initial_uv(
            mesh,
            method="tutte",
            boundary_mode=init_config["boundary"],
            geometry_scale=init_config["geometry_scale"],
        ).uv
        initial_source = "generated_tutte"
    else:
        uv0 = resolve_initial_uv(
            mesh,
            method="tutte",
            boundary_mode=init_config["boundary"],
            initial_uv_path=init_config["initial_uv"],
            prim_path=io_config["prim_path"],
            geometry_scale=init_config["geometry_scale"],
        )
        initial_source = "explicit_initial_uv_file"
    initial_injectivity = validate_uv(uv0, mesh.faces)
    if not initial_injectivity["is_valid"]:
        raise ValueError(
            "mesh PL-NVP requires a legal initial map with no flips or intersections: "
            f"{initial_injectivity}"
        )
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)
    initialization_seconds = time.perf_counter() - total_start
    print(
        f"initial valid=True SD={initial_distortion['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"V={topology['vertex_count']} F={topology['face_count']}"
    )

    training_start = time.perf_counter()
    result = train_mesh_pl_nvp(
        mesh.vertices,
        mesh.faces,
        uv0,
        config,
        initial_model_state=args.initial_model_state,
    )
    training_call_seconds = time.perf_counter() - training_start
    evaluation_start = time.perf_counter()
    uv = result.uv
    final_injectivity = validate_uv(uv, mesh.faces)
    final_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv)
    extended_injectivity = validate_uv_torch(
        result.model_final_uv,
        result.model_faces,
        intersection_batch_size=int(train_config["intersection_batch_size"]),
    )
    if not final_injectivity["is_valid"] or not extended_injectivity["is_valid"]:
        raise RuntimeError(
            "final hard-validity audit failed; no result was saved and no rollback was attempted"
        )

    faces_t = torch.as_tensor(mesh.faces, dtype=torch.long, device=result.model_final_uv.device)
    vertices_t = torch.as_tensor(
        mesh.vertices, dtype=torch.float64, device=result.model_final_uv.device
    )
    per_face_sd = symmetric_dirichlet_per_face(
        vertices_t, faces_t, result.model_final_uv[: len(mesh.vertices)]
    )
    hotspots, vertex_q = q_hotspot_report(
        result.final_diagnostics,
        result.model,
        result.model_initial_uv,
        result.model_final_uv,
        faces_t,
        per_face_sd,
    )
    result.info["q"]["hotspots"] = hotspots
    result.info["final_extended_injectivity"] = extended_injectivity
    result.info["initialization"] = {
        "source": initial_source,
        "method": "tutte" if initial_source == "generated_tutte" else None,
        "boundary": init_config["boundary"],
        "geometry_scale": bool(init_config["geometry_scale"]),
    }

    save_mesh(output, mesh, uv=uv)
    save_uv_plot(output.with_suffix(".initial.uv.png"), uv0, mesh.faces)
    save_uv_plot(output.with_suffix(".uv.png"), uv, mesh.faces)
    save_uv_comparison_plot(
        output.with_suffix(".compare.png"),
        uv0,
        uv,
        mesh.faces,
        initial_title=(
            "Initial UV\n"
            f"SD={initial_distortion['symmetric_dirichlet_area_weighted_mean']:.4g}"
        ),
        final_title=(
            "Mesh PL-NVP (no rollback)\n"
            f"SD={final_distortion['symmetric_dirichlet_area_weighted_mean']:.4g}"
        ),
    )
    save_flip_heatmap(
        output.with_suffix(".initial.flip_heatmap.png"),
        uv0,
        mesh.faces,
        title="Initial UV Flip Heatmap",
    )
    save_flip_heatmap(
        output.with_suffix(".flip_heatmap.png"),
        uv,
        mesh.faces,
        title="Mesh PL-NVP Flip Heatmap",
    )
    save_area_comparison_heatmap(output.with_suffix(".area_compare.png"), uv0, uv, mesh.faces)
    save_distortion_heatmap(
        output.with_suffix(".initial.distortion.png"),
        mesh.vertices,
        mesh.faces,
        uv0,
        title="Initial Symmetric Dirichlet",
    )
    save_distortion_heatmap(
        output.with_suffix(".distortion.png"),
        mesh.vertices,
        mesh.faces,
        uv,
        title="Mesh PL-NVP Symmetric Dirichlet",
    )
    save_distortion_comparison_heatmap(
        output.with_suffix(".distortion_compare.png"),
        mesh.vertices,
        mesh.faces,
        uv0,
        uv,
    )
    save_intersection_heatmap(
        output.with_suffix(".intersection_heatmap.png"),
        uv,
        mesh.faces,
        final_injectivity["intersections"],
    )
    save_loss_plot(output.with_suffix(".loss.png"), result.history)
    plot_q_hotspots(
        result.model_final_uv[: len(mesh.vertices)],
        faces_t,
        vertex_q[: len(mesh.vertices)],
        hotspots,
        output.with_suffix(".q_hotspots.png"),
    )

    if io_config["save_model"]:
        torch.save(result.model.state_dict(), output.with_suffix(".model.pt"))
    evaluation_and_artifact_seconds = time.perf_counter() - evaluation_start
    runtime = {
        "input_and_tutte_seconds": initialization_seconds,
        "training_call_seconds": training_call_seconds,
        "optimization_loop_seconds": float(result.info["elapsed_seconds"]),
        "final_evaluation_and_artifact_seconds": evaluation_and_artifact_seconds,
        "total_seconds_before_metadata_write": time.perf_counter() - total_start,
        "timing_scope": (
            "total includes mesh loading, Tutte, scaffold/model construction, training, "
            "final validity audit, metrics, plots, output mesh, and model save; metadata "
            "JSON/CSV serialization follows this timestamp"
        ),
    }
    result.info["runtime"] = runtime
    with output.with_suffix(".config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    with output.with_suffix(".runtime.json").open("w", encoding="utf-8") as stream:
        json.dump(runtime, stream, indent=2)
    payload = {
        "initial": {
            "injectivity": initial_injectivity,
            "distortion": initial_distortion,
        },
        "final": {
            "injectivity": final_injectivity,
            "distortion": final_distortion,
        },
        "training": result.info,
        "history": result.history,
    }
    with output.with_suffix(".metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    save_run_summary(
        output,
        build_run_summary("mesh_pl_nvp", int(train_config["iters"]), payload),
    )
    print(
        f"final valid=True SD={final_distortion['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"q_max={result.info['q']['all_updates']['max']:.6f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
