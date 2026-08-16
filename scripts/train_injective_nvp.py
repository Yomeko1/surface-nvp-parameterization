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
from surface_nvp.training.config import NVP_OVERRIDE_PATHS, apply_cli_overrides, load_config
from surface_nvp.training.trainer import train_nvp
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot, save_uv_plot
from surface_nvp.visualization.plot_training import save_loss_plot
from surface_nvp.visualization.uv_diagnostics import (
    save_area_comparison_heatmap,
    save_distortion_comparison_heatmap,
    save_flip_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-uv", default=None, help="mesh whose UV coordinates are used as the shared initialization")
    parser.add_argument("--method", choices=["tutte"], default=None)
    parser.add_argument("--boundary", choices=["circle", "square"], default=None)
    parser.add_argument("--coupling-type", choices=["affine", "spline"], default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument("--s-clamp", type=float, default=None)
    parser.add_argument("--spline-bins", type=int, default=None)
    parser.add_argument("--spline-bound", type=float, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--check-interval", type=int, default=None)
    parser.add_argument("--boundary-weight", type=float, default=None)
    parser.add_argument("--identity-weight", type=float, default=None)
    parser.add_argument("--area-weight", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validation-device", default=None, help="defaults to --device; use cpu to force CPU validation")
    parser.add_argument("--intersection-batch-size", type=int, default=None)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args, NVP_OVERRIDE_PATHS)
    init_config = config["init"]
    train_config = config["train"]
    model_config = config["model"]
    io_config = config["io"]

    mesh = load_mesh(args.input, prim_path=io_config["prim_path"])
    uv0 = resolve_initial_uv(
        mesh,
        method=init_config["method"],
        boundary_mode=init_config["boundary"],
        initial_uv_path=init_config["initial_uv"],
        prim_path=io_config["prim_path"],
    )
    initial_metrics = validate_uv(uv0, mesh.faces)
    print("initial", initial_metrics)
    if not initial_metrics["is_valid"]:
        raise ValueError(f"NVP requires a valid initial UV map: {initial_metrics}")
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)

    out = Path(args.output)
    save_uv_plot(out.with_suffix(".initial.uv.png"), uv0, mesh.faces)
    save_flip_heatmap(out.with_suffix(".initial.flip_heatmap.png"), uv0, mesh.faces, title="Initial UV Flip Heatmap")

    uv, _, history, training_info = train_nvp(
        mesh.vertices,
        mesh.faces,
        uv0,
        iters=train_config["iters"],
        lr=train_config["lr"],
        device=train_config["device"],
        check_interval=train_config["check_interval"],
        boundary_weight=train_config["boundary_weight"],
        identity_weight=train_config["identity_weight"],
        area_weight=train_config["area_weight"],
        validation_device=train_config["validation_device"],
        intersection_batch_size=train_config["intersection_batch_size"],
        coupling_type=model_config["coupling_type"],
        num_layers=model_config["num_layers"],
        hidden_dim=model_config["hidden_dim"],
        mlp_layers=model_config["mlp_layers"],
        s_clamp=model_config["s_clamp"],
        spline_bins=model_config["spline_bins"],
        spline_bound=model_config["spline_bound"],
        return_info=True,
    )
    final_metrics = validate_uv(uv, mesh.faces)
    final_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv)
    print("final", final_metrics)
    save_mesh(args.output, mesh, uv=uv)
    save_uv_plot(out.with_suffix(".uv.png"), uv, mesh.faces)
    save_flip_heatmap(out.with_suffix(".flip_heatmap.png"), uv, mesh.faces, title="Final UV Flip Heatmap")
    save_area_comparison_heatmap(out.with_suffix(".area_compare.png"), uv0, uv, mesh.faces)
    save_distortion_comparison_heatmap(out.with_suffix(".distortion_compare.png"), mesh.vertices, mesh.faces, uv0, uv)
    save_uv_comparison_plot(
        out.with_suffix(".compare.png"),
        uv0,
        uv,
        mesh.faces,
        initial_title=f"Initial UV\nvalid={initial_metrics['is_valid']}, SD mean={initial_distortion['symmetric_dirichlet_mean']:.3g}",
        final_title=f"Final UV\nvalid={final_metrics['is_valid']}, SD mean={final_distortion['symmetric_dirichlet_mean']:.3g}",
    )
    save_loss_plot(out.with_suffix(".loss.png"), history)
    with out.with_suffix(".config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with out.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        payload = {
            "initial": {"injectivity": initial_metrics, "distortion": initial_distortion},
            "final": {"injectivity": final_metrics, "distortion": final_distortion},
            "training": training_info,
            "history": history,
        }
        json.dump(payload, f, indent=2)
    save_run_summary(out, build_run_summary(f"nvp_{model_config['coupling_type']}", train_config["iters"], payload))


if __name__ == "__main__":
    main()
