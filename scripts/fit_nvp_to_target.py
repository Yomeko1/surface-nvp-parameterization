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
from surface_nvp.training.supervised import fit_nvp_to_target
from surface_nvp.visualization.plot_training import save_supervised_loss_plot
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot
from surface_nvp.visualization.uv_diagnostics import save_intersection_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an NVP map to a target UV parameterization")
    parser.add_argument("--input", required=True)
    parser.add_argument("--initial-uv", required=True)
    parser.add_argument("--target-uv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coupling-type", choices=["affine", "spline"], required=True)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--s-clamp", type=float, default=2.0)
    parser.add_argument("--spline-bins", type=int, default=8)
    parser.add_argument("--spline-bound", type=float, default=1.1)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()

    mesh = load_mesh(args.input, prim_path=args.prim_path)
    initial_uv = resolve_initial_uv(mesh, initial_uv_path=args.initial_uv, prim_path=args.prim_path)
    target_uv = resolve_initial_uv(mesh, initial_uv_path=args.target_uv, prim_path=args.prim_path)
    target_injectivity = validate_uv(target_uv, mesh.faces)
    if not target_injectivity["is_valid"]:
        raise ValueError(f"target UV must be valid: {target_injectivity}")

    fitted_uv, _, history, fitting = fit_nvp_to_target(
        initial_uv,
        target_uv,
        coupling_type=args.coupling_type,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        mlp_layers=args.mlp_layers,
        s_clamp=args.s_clamp,
        spline_bins=args.spline_bins,
        spline_bound=args.spline_bound,
        iters=args.iters,
        lr=args.lr,
        check_interval=args.check_interval,
        device=args.device,
        seed=args.seed,
    )
    fitted_injectivity = validate_uv(fitted_uv, mesh.faces)
    payload = {
        "config": vars(args),
        "fitting": fitting,
        "initial": {"distortion": compute_distortion_metrics(mesh.vertices, mesh.faces, initial_uv)},
        "target": {
            "injectivity": target_injectivity,
            "distortion": compute_distortion_metrics(mesh.vertices, mesh.faces, target_uv),
        },
        "fitted": {
            "injectivity": fitted_injectivity,
            "distortion": compute_distortion_metrics(mesh.vertices, mesh.faces, fitted_uv),
        },
        "history": history,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_mesh(output, mesh, uv=fitted_uv)
    save_uv_comparison_plot(
        output.with_suffix(".compare.png"),
        target_uv,
        fitted_uv,
        mesh.faces,
        initial_title="SLIM target",
        final_title=f"Fitted {args.coupling_type} NVP",
    )
    save_supervised_loss_plot(output.with_suffix(".loss.png"), history)
    save_intersection_heatmap(
        output.with_suffix(".intersection_heatmap.png"),
        fitted_uv,
        mesh.faces,
        fitted_injectivity["intersections"],
    )
    with output.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"fitting": fitting, "fitted": payload["fitted"]}, indent=2))


if __name__ == "__main__":
    main()
