from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.init_param import resolve_initial_uv
from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.training.metrics import compute_distortion_metrics
from surface_nvp.training.summary import build_run_summary, save_run_summary
from surface_nvp.training.config import COMMON_OVERRIDE_PATHS, apply_cli_overrides, load_config
from surface_nvp.training.trainer import train_direct_uv
from surface_nvp.visualization.plot_training import save_loss_plot
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot, save_uv_plot
from surface_nvp.visualization.uv_diagnostics import (
    save_area_comparison_heatmap,
    save_distortion_comparison_heatmap,
    save_distortion_heatmap,
    save_flip_heatmap,
    save_intersection_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-uv", default=None, help="mesh whose UV coordinates are used as the shared initialization")
    parser.add_argument("--geometry-scale", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--method", choices=["tutte", "mean_value", "abfpp", "auto"], default=None)
    parser.add_argument("--boundary", choices=["circle", "square"], default=None)
    parser.add_argument("--abfpp-executable", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--check-interval", type=int, default=None)
    parser.add_argument("--plateau-patience", type=int, default=None)
    parser.add_argument("--lr-decay", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default=None)
    parser.add_argument("--lbfgs-iters", type=int, default=None)
    parser.add_argument("--lbfgs-lr", type=float, default=None)
    parser.add_argument("--lbfgs-check-interval", type=int, default=None)
    parser.add_argument("--boundary-weight", type=float, default=None, help="deprecated; must be 0")
    parser.add_argument("--identity-weight", type=float, default=None)
    parser.add_argument("--area-weight", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validation-device", default=None)
    parser.add_argument("--intersection-batch-size", type=int, default=None)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args, COMMON_OVERRIDE_PATHS)
    train_config = config["train"]
    mesh = load_mesh(args.input, prim_path=config["io"]["prim_path"])
    uv0 = resolve_initial_uv(
        mesh,
        method=config["init"]["method"],
        boundary_mode=config["init"]["boundary"],
        initial_uv_path=config["init"]["initial_uv"],
        prim_path=config["io"]["prim_path"],
        geometry_scale=config["init"]["geometry_scale"],
        abfpp_executable=config["init"]["abfpp_executable"],
    )
    initial_metrics = validate_uv(uv0, mesh.faces)
    print("initial", initial_metrics)
    if not initial_metrics["is_valid"]:
        raise ValueError(f"Direct UV requires a valid initial UV map: {initial_metrics}")
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)

    uv, history, training_info = train_direct_uv(
        mesh.vertices,
        mesh.faces,
        uv0,
        iters=train_config["iters"],
        lr=train_config["lr"],
        device=train_config["device"],
        check_interval=train_config["check_interval"],
        plateau_patience=train_config["plateau_patience"],
        lr_decay=train_config["lr_decay"],
        min_lr=train_config["min_lr"],
        lr_schedule=train_config["lr_schedule"],
        lbfgs_iters=train_config["lbfgs_iters"],
        lbfgs_lr=train_config["lbfgs_lr"],
        lbfgs_check_interval=train_config["lbfgs_check_interval"],
        boundary_weight=train_config["boundary_weight"],
        identity_weight=train_config["identity_weight"],
        area_weight=train_config["area_weight"],
        validation_device=train_config["validation_device"],
        intersection_batch_size=train_config["intersection_batch_size"],
        seed=train_config["seed"],
    )
    final_metrics = validate_uv(uv, mesh.faces)
    final_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv)
    print("final", final_metrics)

    out = Path(args.output)
    save_mesh(args.output, mesh, uv=uv)
    save_uv_plot(out.with_suffix(".initial.uv.png"), uv0, mesh.faces)
    save_uv_plot(out.with_suffix(".uv.png"), uv, mesh.faces)
    save_flip_heatmap(out.with_suffix(".initial.flip_heatmap.png"), uv0, mesh.faces, title="Initial UV Flip Heatmap")
    save_flip_heatmap(out.with_suffix(".flip_heatmap.png"), uv, mesh.faces, title="Final Direct UV Flip Heatmap")
    save_area_comparison_heatmap(out.with_suffix(".area_compare.png"), uv0, uv, mesh.faces)
    save_distortion_heatmap(
        out.with_suffix(".initial.distortion.png"), mesh.vertices, mesh.faces, uv0, title="Initial Symmetric Dirichlet"
    )
    save_distortion_heatmap(
        out.with_suffix(".distortion.png"), mesh.vertices, mesh.faces, uv, title="Final Direct UV Symmetric Dirichlet"
    )
    save_distortion_comparison_heatmap(out.with_suffix(".distortion_compare.png"), mesh.vertices, mesh.faces, uv0, uv)
    save_intersection_heatmap(
        out.with_suffix(".intersection_heatmap.png"), uv, mesh.faces, final_metrics["intersections"]
    )
    save_uv_comparison_plot(
        out.with_suffix(".compare.png"),
        uv0,
        uv,
        mesh.faces,
        initial_title=f"Initial UV\nvalid={initial_metrics['is_valid']}, SD mean={initial_distortion['symmetric_dirichlet_mean']:.3g}",
        final_title=f"Direct UV\nvalid={final_metrics['is_valid']}, SD mean={final_distortion['symmetric_dirichlet_mean']:.3g}",
    )
    save_loss_plot(out.with_suffix(".loss.png"), history)
    with out.with_suffix(".config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    payload = {
        "initial": {"injectivity": initial_metrics, "distortion": initial_distortion},
        "final": {"injectivity": final_metrics, "distortion": final_distortion},
        "training": training_info,
        "history": history,
    }
    with out.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    total_iters = train_config["iters"] + train_config["lbfgs_iters"]
    save_run_summary(out, build_run_summary("direct_uv", total_iters, payload))


if __name__ == "__main__":
    main()
