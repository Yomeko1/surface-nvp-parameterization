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
    spline_bins: int = 16,
    spline_bound: float = 1.1,
    global_transform: bool = True,
    return_info: bool = False,
    seed: int = 0,
    plateau_patience: int = 4,
    lr_decay: float = 0.5,
    min_lr: float = 1e-6,
    lr_schedule: str = "constant",
    lbfgs_iters: int = 0,
    lbfgs_lr: float = 1.0,
    lbfgs_check_interval: int = 1,
    mixing_type: str = "none",
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
        mixing_type=mixing_type,
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
        lr_schedule=lr_schedule,
        lbfgs_iters=lbfgs_iters,
        lbfgs_lr=lbfgs_lr,
        lbfgs_check_interval=lbfgs_check_interval,
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
    lr_schedule: str = "constant",
    lbfgs_iters: int = 0,
    lbfgs_lr: float = 1.0,
    lbfgs_check_interval: int = 1,
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
        lr_schedule=lr_schedule,
        lbfgs_iters=lbfgs_iters,
        lbfgs_lr=lbfgs_lr,
        lbfgs_check_interval=lbfgs_check_interval,
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
    lr_schedule: str = "constant",
    lbfgs_iters: int = 0,
    lbfgs_lr: float = 1.0,
    lbfgs_check_interval: int = 1,
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
        lr_schedule,
        lbfgs_iters,
        lbfgs_lr,
        lbfgs_check_interval,
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
    lr_multiplier = 1.0
    start_time = time.perf_counter()
    for iteration in range(1, iters + 1):
        scheduled_lr = _scheduled_learning_rate(lr, min_lr, iteration, iters, lr_schedule)
        _set_optimizer_learning_rate(optimizer, max(min_lr, scheduled_lr * lr_multiplier))
        optimizer.zero_grad(set_to_none=True)
        uv = model(uv0_t)
        terms = _loss_terms(
            uv, v_t, f_t, uv0_t, reference_scale, identity_weight, area_weight
        )
        terms["loss"].backward()
        optimizer.step()

        if iteration % check_interval == 0 or iteration == iters:
            uv_np, metrics = _evaluate_parameterization(
                model,
                uv0_t,
                v_t,
                f_t,
                faces,
                reference_scale,
                identity_weight,
                area_weight,
                validation_device,
                device,
                intersection_batch_size,
            )
            metrics.update({"iteration": iteration, "phase": "adam"})
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
                    lr_multiplier *= lr_decay
                    new_lr = max(min_lr, scheduled_lr * lr_multiplier)
                    if current_lr > min_lr and best_safe.restore(model, optimizer):
                        _set_optimizer_learning_rate(optimizer, new_lr)
                        latest_safe.save(model, optimizer, best_safe.iteration, best_safe.uv, best_safe.metrics)
                        metrics["restarted_from_best"] = True
                        metrics["restart_learning_rate"] = new_lr
                        plateau_restarts += 1
                    checks_without_improvement = 0
            else:
                metrics["is_best_valid"] = False
                lr_multiplier *= lr_decay
                restored = latest_safe.restore(model, optimizer)
                if not restored:
                    raise RuntimeError(f"invalid UV at iter {iteration} and no valid checkpoint exists: {metrics}")
                _set_optimizer_learning_rate(optimizer, max(min_lr, scheduled_lr * lr_multiplier))
                invalid_rollbacks += 1
                checks_without_improvement = 0
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            history.append(metrics)

    adam_final_learning_rate = optimizer.param_groups[0]["lr"]
    if best_safe.uv is not None:
        best_safe.restore_model(model)
        best_uv = best_safe.uv
    latest_valid_iteration = int(latest_safe.iteration)

    lbfgs_rollbacks = 0
    lbfgs_optimizer = None
    if lbfgs_iters > 0:
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=1,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )
        lbfgs_latest_safe = SafeCheckpoint()
        lbfgs_latest_safe.save(
            model,
            lbfgs_optimizer,
            best_safe.iteration,
            best_uv,
            best_safe.metrics,
        )
        lbfgs_lr_multiplier = 1.0
        for lbfgs_step in range(1, lbfgs_iters + 1):
            iteration = iters + lbfgs_step
            _set_optimizer_learning_rate(
                lbfgs_optimizer, max(min_lr, lbfgs_lr * lbfgs_lr_multiplier)
            )

            def closure():
                lbfgs_optimizer.zero_grad(set_to_none=True)
                closure_uv = model(uv0_t)
                closure_terms = _loss_terms(
                    closure_uv,
                    v_t,
                    f_t,
                    uv0_t,
                    reference_scale,
                    identity_weight,
                    area_weight,
                )
                closure_terms["loss"].backward()
                return closure_terms["loss"]

            lbfgs_optimizer.step(closure)
            if lbfgs_step % lbfgs_check_interval == 0 or lbfgs_step == lbfgs_iters:
                uv_np, metrics = _evaluate_parameterization(
                    model,
                    uv0_t,
                    v_t,
                    f_t,
                    faces,
                    reference_scale,
                    identity_weight,
                    area_weight,
                    validation_device,
                    device,
                    intersection_batch_size,
                )
                metrics.update({"iteration": iteration, "phase": "lbfgs"})
                if metrics["is_valid"]:
                    metrics["is_best_valid"] = metrics["loss_distortion"] < best_valid_score
                    lbfgs_latest_safe.save(model, lbfgs_optimizer, iteration, uv_np, metrics)
                    latest_valid_iteration = iteration
                    if metrics["is_best_valid"]:
                        best_valid_score = metrics["loss_distortion"]
                        best_safe.save(model, lbfgs_optimizer, iteration, uv_np, metrics)
                        best_uv = uv_np
                else:
                    metrics["is_best_valid"] = False
                    lbfgs_lr_multiplier *= lr_decay
                    restored = lbfgs_latest_safe.restore(model, lbfgs_optimizer)
                    if not restored:
                        raise RuntimeError(
                            f"invalid UV at L-BFGS iter {iteration} and no valid checkpoint exists: {metrics}"
                        )
                    _set_optimizer_learning_rate(
                        lbfgs_optimizer, max(min_lr, lbfgs_lr * lbfgs_lr_multiplier)
                    )
                    lbfgs_rollbacks += 1
                metrics["learning_rate"] = lbfgs_optimizer.param_groups[0]["lr"]
                history.append(metrics)

    final_learning_rate = (
        lbfgs_optimizer.param_groups[0]["lr"]
        if lbfgs_optimizer is not None
        else adam_final_learning_rate
    )
    if best_safe.uv is not None:
        best_safe.restore_model(model)
        best_uv = best_safe.uv
    info = {
        "selected_checkpoint": "best_valid_by_loss_distortion",
        "selected_iteration": int(best_safe.iteration),
        "selected_metrics": dict(best_safe.metrics) if best_safe.metrics is not None else None,
        "latest_valid_iteration": latest_valid_iteration,
        "elapsed_seconds": time.perf_counter() - start_time,
        "plateau_restarts": plateau_restarts,
        "invalid_rollbacks": invalid_rollbacks,
        "lbfgs_rollbacks": lbfgs_rollbacks,
        "adam_final_learning_rate": adam_final_learning_rate,
        "final_learning_rate": final_learning_rate,
        "lr_schedule": lr_schedule,
        "adam_iters": iters,
        "lbfgs_iters": lbfgs_iters,
        "seed": seed,
        "model": {
            "class": type(model).__name__,
            "coupling_type": getattr(model, "coupling_type", None),
            "mixing_type": getattr(model, "mixing_type", None),
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
            "mixing_angles": (
                [float(layer.angle.detach().cpu()) for layer in model.mixing_layers]
                if getattr(model, "mixing_layers", None) is not None
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


def _evaluate_parameterization(
    model,
    uv0_t,
    vertices_t,
    faces_t,
    faces,
    reference_scale,
    identity_weight,
    area_weight,
    validation_device,
    training_device,
    intersection_batch_size,
):
    uv_eval = model(uv0_t).detach()
    if validation_device == training_device:
        metrics = compute_metrics_torch(
            uv_eval,
            faces_t,
            check_intersections=True,
            intersection_batch_size=intersection_batch_size,
        )
        uv_np = uv_eval.cpu().numpy()
    else:
        uv_np = uv_eval.cpu().numpy()
        metrics = compute_metrics(uv_np, faces, check_intersections=True)
    with torch.no_grad():
        eval_terms = _loss_terms(
            uv_eval,
            vertices_t,
            faces_t,
            uv0_t,
            reference_scale,
            identity_weight,
            area_weight,
        )
    metrics.update(_float_loss_terms(eval_terms))
    return uv_np, metrics


def _scheduled_learning_rate(
    initial_lr: float,
    min_lr: float,
    iteration: int,
    total_iterations: int,
    schedule: str,
) -> float:
    if schedule == "constant":
        return initial_lr
    progress = (iteration - 1) / max(total_iterations - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (initial_lr - min_lr) * cosine


def _set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _validate_training_options(
    iters,
    lr,
    check_interval,
    boundary_weight,
    plateau_patience,
    lr_decay,
    min_lr,
    identity_weight,
    area_weight,
    batch_size,
    seed,
    lr_schedule,
    lbfgs_iters,
    lbfgs_lr,
    lbfgs_check_interval,
):
    numeric_values = (
        iters,
        lr,
        check_interval,
        boundary_weight,
        plateau_patience,
        lr_decay,
        min_lr,
        identity_weight,
        area_weight,
        batch_size,
        seed,
        lbfgs_iters,
        lbfgs_lr,
        lbfgs_check_interval,
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("training options must be finite")
    if not all(
        isinstance(value, Integral)
        for value in (
            iters,
            check_interval,
            plateau_patience,
            batch_size,
            seed,
            lbfgs_iters,
            lbfgs_check_interval,
        )
    ):
        raise ValueError(
            "iteration counts, check intervals, plateau_patience, "
            "intersection_batch_size, and seed must be integers"
        )
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
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be 'constant' or 'cosine'")
    if lbfgs_iters < 0:
        raise ValueError("lbfgs_iters must be non-negative")
    if lbfgs_lr <= 0.0:
        raise ValueError("lbfgs_lr must be positive")
    if lbfgs_check_interval <= 0:
        raise ValueError("lbfgs_check_interval must be positive")
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
