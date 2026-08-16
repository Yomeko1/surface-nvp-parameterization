from __future__ import annotations

import copy
import math
import time
from numbers import Integral

import numpy as np
import torch

from surface_nvp.models import NVP2D


def fit_nvp_to_target(
    initial_uv: np.ndarray,
    target_uv: np.ndarray,
    *,
    coupling_type: str = "affine",
    num_layers: int = 6,
    hidden_dim: int = 64,
    mlp_layers: int = 3,
    s_clamp: float = 2.0,
    spline_bins: int = 8,
    spline_bound: float = 1.1,
    iters: int = 5000,
    lr: float = 1e-3,
    check_interval: int = 100,
    device: str = "cpu",
    seed: int = 0,
):
    if initial_uv.shape != target_uv.shape or initial_uv.ndim != 2 or initial_uv.shape[1] != 2:
        raise ValueError("initial_uv and target_uv must have matching [V, 2] shapes")
    numeric_values = (iters, lr, check_interval, seed)
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("supervised fitting options must be finite")
    if not all(isinstance(value, Integral) for value in (iters, check_interval, seed)):
        raise ValueError("iters, check_interval, and seed must be integers")
    if min(iters, check_interval) <= 0 or lr <= 0.0:
        raise ValueError("iters, check_interval, and lr must be positive")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    initial = torch.as_tensor(initial_uv, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_uv, dtype=torch.float32, device=device)
    target_scale = target.std().clamp_min(1e-8)
    model = NVP2D(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        mlp_layers=mlp_layers,
        s_clamp=s_clamp,
        coupling_type=coupling_type,
        spline_bins=spline_bins,
        spline_bound=spline_bound,
    ).to(device)
    model.set_domain(initial)
    if model.global_log_scale is not None:
        initial_extent = (initial.amax(dim=0) - initial.amin(dim=0)).clamp_min(1e-8)
        target_extent = (target.amax(dim=0) - target.amin(dim=0)).clamp_min(1e-8)
        initial_center = 0.5 * (initial.amax(dim=0) + initial.amin(dim=0))
        target_center = 0.5 * (target.amax(dim=0) + target.amin(dim=0))
        with torch.no_grad():
            model.global_log_scale.copy_(torch.log(target_extent / initial_extent))
            model.global_translation.copy_(target_center - initial_center)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=lr * 0.01)

    best_loss = float("inf")
    best_iteration = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    start = time.perf_counter()
    for iteration in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        mapped = model(initial)
        loss = ((mapped - target) / target_scale).pow(2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite supervised loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % check_interval == 0 or iteration == iters:
            with torch.no_grad():
                mapped = model(initial)
                normalized_mse = float((((mapped - target) / target_scale).pow(2).mean()).cpu())
                mse = float((mapped - target).pow(2).mean().cpu())
                max_error = float(torch.linalg.norm(mapped - target, dim=1).max().cpu())
            if normalized_mse < best_loss:
                best_loss = normalized_mse
                best_iteration = iteration
                best_state = copy.deepcopy(model.state_dict())
            history.append(
                {
                    "iteration": iteration,
                    "normalized_mse": normalized_mse,
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "max_vertex_error": max_error,
                    "lr": optimizer.param_groups[0]["lr"],
                    "is_best": iteration == best_iteration,
                }
            )

    model.load_state_dict(best_state)
    with torch.no_grad():
        fitted = model(initial)
        recovered = model.inverse(fitted)
        inverse_max_error = float(torch.abs(recovered - initial).max().cpu())
    return fitted.cpu().numpy(), model, history, {
        "selected_iteration": best_iteration,
        "normalized_mse": best_loss,
        "elapsed_seconds": time.perf_counter() - start,
        "inverse_max_error": inverse_max_error,
        "target_extent": np.ptp(target_uv, axis=0).tolist(),
        "fitted_extent": np.ptp(fitted.cpu().numpy(), axis=0).tolist(),
    }
