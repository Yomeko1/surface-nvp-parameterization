from __future__ import annotations

import math
import time
from numbers import Integral

import numpy as np
import torch
import torch.nn as nn

from surface_nvp.losses.distortion import jacobian_barrier_loss, jacobian_determinants, symmetric_dirichlet_loss
from surface_nvp.losses.regularization import identity_loss
from surface_nvp.models import DirectUV, NVP2D
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
    boundary_weight: float = 0.0,
    identity_weight: float = 1e-3,
    area_weight: float = 100.0,
    validation_device: str | None = None,
    intersection_batch_size: int = 262144,
    coupling_type: str = "affine",
    num_layers: int = 6,
    hidden_dim: int = 64,
    mlp_layers: int = 3,
    s_clamp: float = 2.0,
    spline_bins: int = 8,
    spline_bound: float = 1.1,
    global_transform: bool = False,
    return_info: bool = False,
    seed: int = 0,
    plateau_patience: int = 4,
    lr_decay: float = 0.5,
    min_lr: float = 1e-6,
):
    _set_seed(seed)
    model = NVP2D(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        mlp_layers=mlp_layers,
        s_clamp=s_clamp,
        coupling_type=coupling_type,
        spline_bins=spline_bins,
        spline_bound=spline_bound,
        global_transform=global_transform,
    )
    result = train_parameterization(
        model,
        vertices,
        faces,
        uv0,
        iters=iters,
        lr=lr,
        device=device,
        check_interval=check_interval,
        boundary_weight=boundary_weight,
        plateau_patience=plateau_patience,
        lr_decay=lr_decay,
        min_lr=min_lr,
        identity_weight=identity_weight,
        area_weight=area_weight,
        validation_device=validation_device,
        intersection_batch_size=intersection_batch_size,
        seed=seed,
    )
    best_uv, model, history, info = result
    if return_info:
        return best_uv, model, history, info
    return best_uv, model, history


def train_direct_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv0: np.ndarray,
    iters: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
    check_interval: int = 25,
    boundary_weight: float = 0.0,
    identity_weight: float = 1e-3,
    area_weight: float = 100.0,
    validation_device: str | None = None,
    intersection_batch_size: int = 262144,
    seed: int = 0,
    plateau_patience: int = 4,
    lr_decay: float = 0.5,
    min_lr: float = 1e-6,
):
    _set_seed(seed)
    model = DirectUV(torch.as_tensor(uv0, dtype=torch.float32))
    best_uv, _, history, info = train_parameterization(
        model,
        vertices,
        faces,
        uv0,
        iters=iters,
        lr=lr,
        device=device,
        check_interval=check_interval,
        boundary_weight=boundary_weight,
        plateau_patience=plateau_patience,
        lr_decay=lr_decay,
        min_lr=min_lr,
        identity_weight=identity_weight,
        area_weight=area_weight,
        validation_device=validation_device,
        intersection_batch_size=intersection_batch_size,
        seed=seed,
    )
    return best_uv, history, info


def train_parameterization(
    model: nn.Module,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv0: np.ndarray,
    iters: int,
    lr: float,
    device: str,
    check_interval: int,
    boundary_weight: float,
    identity_weight: float,
    area_weight: float,
    validation_device: str | None,
    intersection_batch_size: int,
    seed: int = 0,
    plateau_patience: int = 4,
    lr_decay: float = 0.5,
    min_lr: float = 1e-6,
):
    _validate_training_options(
        iters,
        lr,
        check_interval,
        boundary_weight,
        plateau_patience,
        lr_decay,
        min_lr,
        identity_weight,
        area_weight,
        intersection_batch_size,
        seed,
    )
    model = model.to(device)
    v_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    f_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    uv0_t = torch.as_tensor(uv0, dtype=torch.float32, device=device)
    if hasattr(model, "set_domain"):
        model.set_domain(uv0_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    validation_device = validation_device or device
    with torch.no_grad():
        initial_det = jacobian_determinants(v_t, f_t, uv0_t)
        reference_scale = initial_det.abs().median().clamp_min(1e-12)
    latest_safe = SafeCheckpoint()
    best_safe = SafeCheckpoint()

    with torch.no_grad():
        initial_model_uv = model(uv0_t)
    best_uv = initial_model_uv.detach().cpu().numpy()
    init_metrics = compute_metrics(best_uv, faces, check_intersections=True)
    if not init_metrics["is_valid"]:
        raise ValueError(f"optimization requires a valid initial UV map: {init_metrics}")
    with torch.no_grad():
        init_terms = _loss_terms(
            initial_model_uv, v_t, f_t, uv0_t, reference_scale, identity_weight, area_weight
        )
    init_metrics.update({"iteration": 0, **_float_loss_terms(init_terms)})
    latest_safe.save(model, optimizer, 0, best_uv, init_metrics)
    best_valid_score = init_metrics["loss_distortion"]
    best_safe.save(model, optimizer, 0, best_uv, init_metrics)

    history = []
    checks_without_improvement = 0
    plateau_restarts = 0
    invalid_rollbacks = 0
    start_time = time.perf_counter()
    for iteration in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        uv = model(uv0_t)
        terms = _loss_terms(
            uv, v_t, f_t, uv0_t, reference_scale, identity_weight, area_weight
        )
        terms["loss"].backward()
        optimizer.step()

        if iteration % check_interval == 0 or iteration == iters:
            uv_eval = model(uv0_t).detach()
            if validation_device == device:
                metrics = compute_metrics_torch(uv_eval, f_t, check_intersections=True, intersection_batch_size=intersection_batch_size)
                uv_np = uv_eval.cpu().numpy()
            else:
                uv_np = uv_eval.cpu().numpy()
                metrics = compute_metrics(uv_np, faces, check_intersections=True)
            with torch.no_grad():
                eval_terms = _loss_terms(
                    uv_eval, v_t, f_t, uv0_t, reference_scale, identity_weight, area_weight
                )
            metrics.update({"iteration": iteration, **_float_loss_terms(eval_terms)})
            if metrics["is_valid"]:
                latest_safe.save(model, optimizer, iteration, uv_np, metrics)
                score = metrics["loss_distortion"]
                if score < best_valid_score:
                    best_valid_score = score
                    metrics["is_best_valid"] = True
                    best_safe.save(model, optimizer, iteration, uv_np, metrics)
                    checks_without_improvement = 0
                else:
                    metrics["is_best_valid"] = False
                    checks_without_improvement += 1
                if checks_without_improvement >= plateau_patience:
                    current_lr = optimizer.param_groups[0]["lr"]
                    new_lr = max(current_lr * lr_decay, min_lr)
                    if current_lr > min_lr and best_safe.restore(model, optimizer):
                        for group in optimizer.param_groups:
                            group["lr"] = new_lr
                        latest_safe.save(model, optimizer, best_safe.iteration, best_safe.uv, best_safe.metrics)
                        metrics["restarted_from_best"] = True
                        metrics["restart_learning_rate"] = new_lr
                        plateau_restarts += 1
                    checks_without_improvement = 0
            else:
                metrics["is_best_valid"] = False
                restored = latest_safe.restore(model, optimizer)
                if not restored:
                    raise RuntimeError(f"invalid UV at iter {iteration} and no valid checkpoint exists: {metrics}")
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * lr_decay, min_lr)
                invalid_rollbacks += 1
                checks_without_improvement = 0
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            history.append(metrics)

    final_learning_rate = optimizer.param_groups[0]["lr"]
    if best_safe.uv is not None:
        best_safe.restore(model, optimizer)
        best_uv = best_safe.uv
    info = {
        "selected_checkpoint": "best_valid_by_loss_distortion",
        "selected_iteration": int(best_safe.iteration),
        "selected_metrics": dict(best_safe.metrics) if best_safe.metrics is not None else None,
        "latest_valid_iteration": int(latest_safe.iteration),
        "elapsed_seconds": time.perf_counter() - start_time,
        "plateau_restarts": plateau_restarts,
        "invalid_rollbacks": invalid_rollbacks,
        "final_learning_rate": final_learning_rate,
        "seed": seed,
        "model": {
            "class": type(model).__name__,
            "coupling_type": getattr(model, "coupling_type", None),
            "global_scale": (
                torch.exp(model.global_log_scale).detach().cpu().tolist()
                if getattr(model, "global_log_scale", None) is not None
                else None
            ),
            "global_translation": (
                model.global_translation.detach().cpu().tolist()
                if getattr(model, "global_translation", None) is not None
                else None
            ),
        },
        "objective": {
            "distortion": "3d_face_area_weighted_symmetric_dirichlet",
            "injectivity": "normalized_jacobian_soft_barrier_plus_discrete_validation",
            "jacobian_reference_scale": float(reference_scale.detach().cpu()),
        },
    }
    return best_uv, model, history, info


def _loss_terms(model_uv, vertices, faces, uv0, reference_scale, identity_weight, area_weight):
    loss_dist = symmetric_dirichlet_loss(vertices, faces, model_uv)
    loss_identity = identity_loss(model_uv, uv0)
    loss_jacobian = jacobian_barrier_loss(vertices, faces, model_uv, reference_scale)
    return {
        "loss": loss_dist + identity_weight * loss_identity + area_weight * loss_jacobian,
        "loss_distortion": loss_dist,
        "loss_identity": loss_identity,
        "loss_jacobian": loss_jacobian,
        "weighted_loss_identity": identity_weight * loss_identity,
        "weighted_loss_jacobian": area_weight * loss_jacobian,
    }


def _float_loss_terms(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    values = {key: float(value.detach().cpu()) for key, value in terms.items()}
    values["loss_area"] = values["loss_jacobian"]
    values["weighted_loss_area"] = values["weighted_loss_jacobian"]
    return values


def _validate_training_options(
    iters, lr, check_interval, boundary_weight, plateau_patience, lr_decay, min_lr, identity_weight, area_weight, batch_size, seed
):
    numeric_values = (
        iters, lr, check_interval, boundary_weight, plateau_patience, lr_decay, min_lr, identity_weight, area_weight, batch_size, seed
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("training options must be finite")
    if not all(isinstance(value, Integral) for value in (iters, check_interval, plateau_patience, batch_size, seed)):
        raise ValueError("iters, check_interval, plateau_patience, intersection_batch_size, and seed must be integers")
    if iters <= 0:
        raise ValueError("iters must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if check_interval <= 0:
        raise ValueError("check_interval must be positive")
    if boundary_weight != 0.0:
        raise ValueError("boundary_weight was removed in v2.1; set it to 0")
    if plateau_patience <= 0:
        raise ValueError("plateau_patience must be positive")
    if not 0.0 < lr_decay < 1.0:
        raise ValueError("lr_decay must be between 0 and 1")
    if min_lr <= 0.0 or min_lr > lr:
        raise ValueError("min_lr must be positive and no greater than lr")
    if min(identity_weight, area_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if batch_size <= 0:
        raise ValueError("intersection_batch_size must be positive")


def _set_seed(seed: int) -> None:
    if not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
