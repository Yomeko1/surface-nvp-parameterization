from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.init_param import tutte_parameterize
from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.training.metrics import compute_distortion_metrics
from surface_nvp.training.summary import build_run_summary, save_run_summary
from surface_nvp.training.trainer import train_nvp
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot, save_uv_plot
from surface_nvp.visualization.plot_training import save_loss_plot
from surface_nvp.visualization.uv_diagnostics import (
    save_area_comparison_heatmap,
    save_distortion_comparison_heatmap,
    save_flip_heatmap,
)


DEFAULT_CONFIG = {
    "init": {"method": "tutte", "boundary": "circle"},
    "train": {
        "iters": 1000,
        "lr": 1e-3,
        "check_interval": 25,
        "boundary_weight": 10.0,
        "identity_weight": 1e-3,
        "area_weight": 100.0,
        "device": "cpu",
        "validation_device": None,
        "intersection_batch_size": 262144,
    },
    "io": {"prim_path": None},
}


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_config(path: str | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a YAML mapping")
    return _deep_update(config, loaded)


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "method": ("init", "method"),
        "boundary": ("init", "boundary"),
        "iters": ("train", "iters"),
        "lr": ("train", "lr"),
        "check_interval": ("train", "check_interval"),
        "boundary_weight": ("train", "boundary_weight"),
        "identity_weight": ("train", "identity_weight"),
        "area_weight": ("train", "area_weight"),
        "device": ("train", "device"),
        "validation_device": ("train", "validation_device"),
        "intersection_batch_size": ("train", "intersection_batch_size"),
        "prim_path": ("io", "prim_path"),
    }
    for attr, path in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            config[path[0]][path[1]] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["tutte"], default=None)
    parser.add_argument("--boundary", choices=["circle", "square"], default=None)
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
    config = _apply_cli_overrides(_load_config(args.config), args)
    init_config = config["init"]
    train_config = config["train"]
    io_config = config["io"]

    mesh = load_mesh(args.input, prim_path=io_config["prim_path"])
    if mesh.uv is not None:
        uv0 = mesh.uv
    elif init_config["method"] == "tutte":
        uv0 = tutte_parameterize(mesh.vertices, mesh.faces, boundary_mode=init_config["boundary"])
    else:
        raise ValueError(f"unsupported init method: {init_config['method']}")
    initial_metrics = validate_uv(uv0, mesh.faces)
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)
    print("initial", initial_metrics)

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
    save_run_summary(out, build_run_summary("nvp", train_config["iters"], payload))


if __name__ == "__main__":
    main()
