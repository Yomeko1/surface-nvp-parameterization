from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.geometry.boundary import extract_boundary_loop
from surface_nvp.init_param import tutte_parameterize
from surface_nvp.injectivity.validators import validate_uv, validate_uv_torch
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.losses.area_barrier import area_barrier_loss
from surface_nvp.losses.boundary import boundary_position_loss
from surface_nvp.losses.distortion import symmetric_dirichlet_loss
from surface_nvp.losses.regularization import identity_loss
from surface_nvp.training.metrics import compute_distortion_metrics
from surface_nvp.training.summary import build_run_summary, save_run_summary
from surface_nvp.visualization.plot_training import save_loss_plot
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot, save_uv_plot
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
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--check-interval", type=int, default=None)
    parser.add_argument("--boundary-weight", type=float, default=None)
    parser.add_argument("--identity-weight", type=float, default=None)
    parser.add_argument("--area-weight", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validation-device", default=None)
    parser.add_argument("--intersection-batch-size", type=int, default=None)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()

    config = _apply_cli_overrides(_load_config(args.config), args)
    train_config = config["train"]
    mesh = load_mesh(args.input, prim_path=config["io"]["prim_path"])
    uv0 = mesh.uv if mesh.uv is not None else tutte_parameterize(mesh.vertices, mesh.faces, boundary_mode=config["init"]["boundary"])
    initial_metrics = validate_uv(uv0, mesh.faces)
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)
    print("initial", initial_metrics)

    uv, history, training_info = train_direct_uv(
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
    save_distortion_comparison_heatmap(out.with_suffix(".distortion_compare.png"), mesh.vertices, mesh.faces, uv0, uv)
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
    save_run_summary(out, build_run_summary("direct_uv", train_config["iters"], payload))


def train_direct_uv(
    vertices,
    faces,
    uv0,
    iters: int,
    lr: float,
    device: str,
    check_interval: int,
    boundary_weight: float,
    identity_weight: float,
    area_weight: float,
    validation_device: str | None,
    intersection_batch_size: int,
):
    v_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    f_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    uv0_t = torch.as_tensor(uv0, dtype=torch.float32, device=device)
    uv_param = torch.nn.Parameter(uv0_t.clone())
    optimizer = torch.optim.Adam([uv_param], lr=lr)
    boundary = torch.as_tensor(extract_boundary_loop(faces), dtype=torch.long, device=device)
    validation_device = validation_device or device

    latest_valid_uv = uv0.copy()
    latest_valid_iteration = 0
    best_valid_uv = uv0.copy()
    with torch.no_grad():
        best_valid_score = float(symmetric_dirichlet_loss(v_t, f_t, uv0_t).detach().cpu())
    selected_metrics = {"iteration": 0, "loss_distortion": best_valid_score}
    history = []

    for iteration in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_dist = symmetric_dirichlet_loss(v_t, f_t, uv_param)
        loss_boundary = boundary_position_loss(uv_param, uv0_t, boundary)
        loss_identity = identity_loss(uv_param, uv0_t)
        loss_area = area_barrier_loss(uv_param, f_t)
        loss = loss_dist + boundary_weight * loss_boundary + identity_weight * loss_identity + area_weight * loss_area
        loss.backward()
        optimizer.step()

        if iteration % check_interval == 0 or iteration == iters:
            uv_eval = uv_param.detach()
            if validation_device == device:
                metrics = validate_uv_torch(uv_eval, f_t, check_intersections=True, intersection_batch_size=intersection_batch_size)
            else:
                metrics = validate_uv(uv_eval.cpu().numpy(), faces, check_intersections=True)
            metrics.update(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "loss_distortion": float(loss_dist.detach().cpu()),
                    "loss_boundary": float(loss_boundary.detach().cpu()),
                    "loss_identity": float(loss_identity.detach().cpu()),
                    "loss_area": float(loss_area.detach().cpu()),
                    "weighted_loss_boundary": float((boundary_weight * loss_boundary).detach().cpu()),
                    "weighted_loss_identity": float((identity_weight * loss_identity).detach().cpu()),
                    "weighted_loss_area": float((area_weight * loss_area).detach().cpu()),
                }
            )
            if metrics["is_valid"]:
                latest_valid_uv = uv_eval.cpu().numpy()
                latest_valid_iteration = iteration
                if metrics["loss_distortion"] < best_valid_score:
                    best_valid_score = metrics["loss_distortion"]
                    best_valid_uv = latest_valid_uv.copy()
                    selected_metrics = dict(metrics)
                    metrics["is_best_valid"] = True
                else:
                    metrics["is_best_valid"] = False
            else:
                metrics["is_best_valid"] = False
                with torch.no_grad():
                    uv_param.copy_(torch.as_tensor(latest_valid_uv, dtype=torch.float32, device=device))
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
            history.append(metrics)

    return best_valid_uv, history, {
        "selected_checkpoint": "best_valid_by_loss_distortion",
        "selected_iteration": int(selected_metrics.get("iteration", 0)),
        "selected_metrics": selected_metrics,
        "latest_valid_iteration": int(latest_valid_iteration),
    }


if __name__ == "__main__":
    main()
