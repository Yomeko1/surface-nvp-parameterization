from __future__ import annotations

import numpy as np
import torch

from surface_nvp.geometry.boundary import extract_boundary_loop
from surface_nvp.losses.area_barrier import area_barrier_loss
from surface_nvp.losses.boundary import boundary_position_loss
from surface_nvp.losses.distortion import symmetric_dirichlet_loss
from surface_nvp.losses.regularization import identity_loss
from surface_nvp.models import NVP2D
from surface_nvp.training.metrics import compute_metrics, compute_metrics_torch
from surface_nvp.training.rollback import SafeCheckpoint


def train_nvp(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv0: np.ndarray,
    iters: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
    check_interval: int = 25,
    boundary_weight: float = 10.0,
    identity_weight: float = 1e-3,
    area_weight: float = 100.0,
    validation_device: str | None = None,
    intersection_batch_size: int = 262144,
    return_info: bool = False,
):
    model = NVP2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    v_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    f_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    uv0_t = torch.as_tensor(uv0, dtype=torch.float32, device=device)
    boundary = torch.as_tensor(extract_boundary_loop(faces), dtype=torch.long, device=device)
    validation_device = validation_device or device
    latest_safe = SafeCheckpoint()
    best_safe = SafeCheckpoint()
    best_valid_score = float("inf")

    best_uv = uv0.copy()
    init_metrics = compute_metrics(best_uv, faces, check_intersections=True)
    if init_metrics["is_valid"]:
        with torch.no_grad():
            init_score = float(symmetric_dirichlet_loss(v_t, f_t, uv0_t).detach().cpu())
        init_metrics["iteration"] = 0
        init_metrics["loss_distortion"] = init_score
        latest_safe.save(model, optimizer, 0, best_uv, init_metrics)
        best_valid_score = init_score
        best_safe.save(model, optimizer, 0, best_uv, init_metrics)

    history = []
    for iteration in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        uv = model(uv0_t)
        loss_dist = symmetric_dirichlet_loss(v_t, f_t, uv)
        loss_boundary = boundary_position_loss(uv, uv0_t, boundary)
        loss_identity = identity_loss(uv, uv0_t)
        loss_area = area_barrier_loss(uv, f_t)
        loss = loss_dist + boundary_weight * loss_boundary + identity_weight * loss_identity + area_weight * loss_area
        loss.backward()
        optimizer.step()

        if iteration % check_interval == 0 or iteration == iters:
            uv_eval = model(uv0_t).detach()
            if validation_device == device:
                metrics = compute_metrics_torch(uv_eval, f_t, check_intersections=True, intersection_batch_size=intersection_batch_size)
                uv_np = uv_eval.cpu().numpy()
            else:
                uv_np = uv_eval.cpu().numpy()
                metrics = compute_metrics(uv_np, faces, check_intersections=True)
            metrics["iteration"] = iteration
            metrics["loss"] = float(loss.detach().cpu())
            metrics["loss_distortion"] = float(loss_dist.detach().cpu())
            metrics["loss_boundary"] = float(loss_boundary.detach().cpu())
            metrics["loss_identity"] = float(loss_identity.detach().cpu())
            metrics["loss_area"] = float(loss_area.detach().cpu())
            metrics["weighted_loss_boundary"] = float((boundary_weight * loss_boundary).detach().cpu())
            metrics["weighted_loss_identity"] = float((identity_weight * loss_identity).detach().cpu())
            metrics["weighted_loss_area"] = float((area_weight * loss_area).detach().cpu())
            if metrics["is_valid"]:
                latest_safe.save(model, optimizer, iteration, uv_np, metrics)
                best_uv = uv_np
                score = metrics.get("loss_distortion", metrics.get("loss", float("inf")))
                if score < best_valid_score:
                    best_valid_score = score
                    metrics["is_best_valid"] = True
                    best_safe.save(model, optimizer, iteration, uv_np, metrics)
                else:
                    metrics["is_best_valid"] = False
            else:
                metrics["is_best_valid"] = False
                restored = latest_safe.restore(model, optimizer)
                if not restored:
                    raise RuntimeError(f"invalid UV at iter {iteration} and no valid checkpoint exists: {metrics}")
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
            history.append(metrics)

    if best_safe.uv is not None:
        best_safe.restore(model, optimizer)
        best_uv = best_safe.uv
    info = {
        "selected_checkpoint": "best_valid_by_loss_distortion",
        "selected_iteration": int(best_safe.iteration),
        "selected_metrics": dict(best_safe.metrics) if best_safe.metrics is not None else None,
        "latest_valid_iteration": int(latest_safe.iteration),
    }
    if return_info:
        return best_uv, model, history, info
    return best_uv, model, history
