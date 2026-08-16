from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.init_param import resolve_initial_uv
from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.training.metrics import compute_distortion_metrics
from surface_nvp.training.summary import build_run_summary, save_run_summary
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--executable", required=True, help="path to the surface_nvp_slim executable")
    parser.add_argument("--initial-uv", default=None, help="mesh whose UV coordinates are used as the shared initialization")
    parser.add_argument("--geometry-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--boundary", choices=["circle", "square"], default="circle")
    parser.add_argument("--prim-path", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    if args.iters <= 0:
        parser.error("--iters must be positive")
    executable = Path(args.executable).resolve()
    if not executable.is_file():
        parser.error(f"SLIM executable does not exist: {executable}")

    mesh = load_mesh(args.input, prim_path=args.prim_path)
    uv0 = resolve_initial_uv(
        mesh,
        boundary_mode=args.boundary,
        initial_uv_path=args.initial_uv,
        prim_path=args.prim_path,
        geometry_scale=args.geometry_scale,
    )
    initial_metrics = validate_uv(uv0, mesh.faces)
    if not initial_metrics["is_valid"]:
        raise ValueError(f"SLIM requires a valid initial UV map: {initial_metrics}")
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)

    with tempfile.TemporaryDirectory(prefix="surface_nvp_slim_") as temp_dir:
        temp_dir = Path(temp_dir)
        slim_input = temp_dir / "input.obj"
        slim_output = temp_dir / "output.obj"
        save_mesh(slim_input, mesh, uv=uv0)
        start = time.perf_counter()
        result = subprocess.run(
            [str(executable), str(slim_input), str(slim_output), str(args.iters)],
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed_seconds = time.perf_counter() - start
        if result.returncode != 0:
            raise RuntimeError(
                f"SLIM failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        slim_mesh = load_mesh(slim_output)
        if slim_mesh.uv is None:
            raise RuntimeError("SLIM output contains no UV coordinates")
        if slim_mesh.faces.shape != mesh.faces.shape or (slim_mesh.faces != mesh.faces).any():
            raise RuntimeError("SLIM output topology differs from the input mesh")
        uv = slim_mesh.uv

    final_metrics = validate_uv(uv, mesh.faces)
    final_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv)
    out = Path(args.output)
    save_mesh(out, mesh, uv=uv)
    save_uv_plot(out.with_suffix(".initial.uv.png"), uv0, mesh.faces)
    save_uv_plot(out.with_suffix(".uv.png"), uv, mesh.faces)
    save_flip_heatmap(out.with_suffix(".initial.flip_heatmap.png"), uv0, mesh.faces, title="Initial UV Flip Heatmap")
    save_flip_heatmap(out.with_suffix(".flip_heatmap.png"), uv, mesh.faces, title="Final SLIM UV Flip Heatmap")
    save_area_comparison_heatmap(out.with_suffix(".area_compare.png"), uv0, uv, mesh.faces)
    save_distortion_heatmap(
        out.with_suffix(".initial.distortion.png"), mesh.vertices, mesh.faces, uv0, title="Initial Symmetric Dirichlet"
    )
    save_distortion_heatmap(
        out.with_suffix(".distortion.png"), mesh.vertices, mesh.faces, uv, title="Final SLIM Symmetric Dirichlet"
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
        final_title=f"SLIM\nvalid={final_metrics['is_valid']}, SD mean={final_distortion['symmetric_dirichlet_mean']:.3g}",
    )

    training = {
        "selected_checkpoint": "slim_final_iteration",
        "selected_iteration": args.iters,
        "elapsed_seconds": elapsed_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    payload = {
        "initial": {"injectivity": initial_metrics, "distortion": initial_distortion},
        "final": {"injectivity": final_metrics, "distortion": final_distortion},
        "training": training,
        "history": [],
    }
    with out.with_suffix(".config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with out.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_run_summary(out, build_run_summary("slim", args.iters, payload))
    print("initial", initial_metrics)
    print("final", final_metrics)
    if not final_metrics["is_valid"]:
        raise RuntimeError(f"SLIM produced an invalid UV map: {final_metrics}")


if __name__ == "__main__":
    main()
