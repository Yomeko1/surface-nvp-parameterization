"""Run two- and three-stage certified harmonic/local PL-NVP experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from surface_nvp.init_param import generate_initial_uv
from surface_nvp.injectivity.validators import validate_uv, validate_uv_torch
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.losses.distortion import symmetric_dirichlet_loss
from surface_nvp.training.metrics import compute_distortion_metrics
from surface_nvp.training.summary import build_run_summary, save_run_summary
from surface_nvp.visualization.plot_training import save_loss_plot
from surface_nvp.visualization.plot_uv import save_uv_comparison_plot, save_uv_plot
from surface_nvp.visualization.uv_diagnostics import (
    save_distortion_comparison_heatmap,
    save_flip_heatmap,
    save_intersection_heatmap,
)

from .harmonic_local import HarmonicLocalHarmonicFlow
from .harmonic_modes import (
    build_piecewise_linear_boundary_harmonic_modes,
    build_radial_fourier_harmonic_modes,
    preflight_mode_intervals,
    select_supported_boundary_hat_counts,
)
from .mesh_coupling import signed_double_areas
from .multiring_scaffold import (
    build_multiring_outer_scaffold,
    multiring_transition_profile,
)
from .pipeline_training import validate_disk_topology
from .radial_polytope import radial_conditioning_risk
from .risk_recovery import RiskLearningRateRecovery
from .run_harmonic_modes import _boundary_summary, save_boundary_comparison
from .scaffold import build_outer_scaffold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certified v3.1 H1 -> local spline PL-NVP -> H2 pipeline"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional flat YAML defaults; explicit CLI options take precedence",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prim-path", default=None)
    parser.add_argument("--frequencies", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument(
        "--boundary-hat-counts",
        type=int,
        nargs="*",
        default=[4, 8, 16, 32],
        help="multiscale counts of C0 piecewise-linear boundary controls",
    )
    parser.add_argument(
        "--auto-boundary-hat-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop requested hat scales with insufficient sampled boundary support",
    )
    parser.add_argument(
        "--boundary-hat-min-peak-weight",
        type=float,
        default=0.05,
        help="minimum sampled peak required for every hat in an automatic scale",
    )
    parser.add_argument("--harmonic-cycles", type=int, default=2)
    parser.add_argument("--harmonic-iters", type=int, default=100)
    parser.add_argument("--local-iters", type=int, default=500)
    parser.add_argument("--final-harmonic-iters", type=int, default=100)
    parser.add_argument("--harmonic-lr", type=float, default=0.02)
    parser.add_argument("--local-lr", type=float, default=0.003)
    parser.add_argument(
        "--local-risk-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local-risk-threshold", type=float, default=0.85)
    parser.add_argument("--local-risk-floor-lr", type=float, default=1.0e-6)
    parser.add_argument("--local-risk-global-factor", type=float, default=0.25)
    parser.add_argument("--local-risk-warmup-iters", type=int, default=100)
    parser.add_argument("--local-risk-warmup-start-factor", type=float, default=0.25)
    parser.add_argument("--local-dynamic-risk-threshold", type=float, default=0.95)
    parser.add_argument("--local-dynamic-floor-lr", type=float, default=1.0e-8)
    parser.add_argument(
        "--local-risk-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cautiously recover local LR after sustained conditioning-risk headroom",
    )
    parser.add_argument("--local-risk-recovery-threshold", type=float, default=0.94)
    parser.add_argument("--local-risk-recovery-patience", type=int, default=20)
    parser.add_argument("--local-risk-recovery-interval", type=int, default=10)
    parser.add_argument("--local-risk-recovery-factor", type=float, default=1.05)
    parser.add_argument("--harmonic-max-log-scale", type=float, default=0.08)
    parser.add_argument("--harmonic-max-shift", type=float, default=0.20)
    parser.add_argument("--final-harmonic-max-log-scale", type=float, default=None)
    parser.add_argument("--final-harmonic-max-shift", type=float, default=None)
    parser.add_argument("--area-margin-ratio", type=float, default=1.0e-6)
    parser.add_argument("--local-cycles", type=int, default=4)
    parser.add_argument("--local-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--local-radial-map", choices=("softsign", "atanh"), default="atanh"
    )
    parser.add_argument(
        "--local-latent-transform",
        choices=("affine", "spline"),
        default="spline",
    )
    parser.add_argument("--local-spline-bins", type=int, default=8)
    parser.add_argument("--local-spline-bound", type=float, default=8.0)
    parser.add_argument("--local-max-log-scale", type=float, default=0.08)
    parser.add_argument("--local-max-shift-fraction", type=float, default=0.04)
    parser.add_argument("--local-center-iterations", type=int, default=12)
    parser.add_argument("--scaffold-scale", type=float, default=1.1)
    parser.add_argument("--scaffold-rings", type=int, default=6)
    parser.add_argument("--scaffold-transition-exponent", type=float, default=3.0)
    parser.add_argument(
        "--scaffold-mode-profile",
        choices=("harmonic", "geometric"),
        default="geometric",
        help="how global modes decay across an opt-in multiring scaffold",
    )
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--intersection-batch-size", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    return parser


def parse_args(argv: list[str] | None = None, *, parser=None) -> argparse.Namespace:
    """Load optional YAML defaults before applying explicit CLI overrides."""

    parser = build_parser() if parser is None else parser
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    known, _ = config_probe.parse_known_args(argv)
    if known.config is not None:
        with Path(known.config).open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError("v3.1 config file must contain a flat YAML mapping")
        forbidden = {"help", "config", "input", "output_dir", "prim_path"}
        actions = {action.dest: action for action in parser._actions}
        allowed = set(actions) - forbidden
        unknown = sorted(set(loaded) - allowed)
        if unknown:
            raise ValueError(f"unknown v3.1 config keys: {', '.join(unknown)}")
        validated: dict[str, Any] = {}
        for key, value in loaded.items():
            action = actions[key]
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    raise ValueError(f"v3.1 config key {key} must be boolean")
                converted = value
            elif action.nargs in {"+", "*"}:
                if not isinstance(value, list):
                    raise ValueError(f"v3.1 config key {key} must be a list")
                converted = [action.type(item) for item in value]
            elif action.type is not None:
                converted = action.type(value)
            else:
                converted = value
            if action.choices is not None and converted not in action.choices:
                choices = ", ".join(map(str, action.choices))
                raise ValueError(
                    f"v3.1 config key {key} must be one of: {choices}"
                )
            validated[key] = converted
        parser.set_defaults(**validated)
    return parser.parse_args(argv)


def _train_harmonic_stage(
    module,
    base_uv,
    vertices,
    faces,
    reference_uv,
    boundary,
    *,
    iterations: int,
    learning_rate: float,
    check_interval: int,
    gradient_clip: float,
    phase: str,
    iteration_offset: int,
    observer=None,
    resume_state=None,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    start_iteration = 0 if resume_state is None else int(resume_state["step"])
    if not 0 <= start_iteration <= iterations:
        raise ValueError("invalid harmonic resume step")
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    final = None
    for stage_iteration in range(start_iteration, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        mapped, diagnostics = module(base_uv, return_diagnostics=True)
        loss = symmetric_dirichlet_loss(
            vertices,
            faces,
            mapped[: vertices.shape[0]],
        )
        areas = signed_double_areas(mapped, module.faces)
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"non-finite loss in {phase} at {stage_iteration}")
        if not bool(torch.all(areas.detach() > module.area_floor.detach())):
            raise RuntimeError(
                f"hard area assertion failed in {phase} at {stage_iteration}; no rollback"
            )
        if observer is not None:
            observer(stage_iteration, mapped, loss, diagnostics, optimizer)
        if stage_iteration % check_interval == 0 or stage_iteration == iterations:
            motion = _boundary_summary(reference_uv, mapped.detach(), boundary)
            history.append(
                {
                    "iteration": iteration_offset + stage_iteration,
                    "stage_iteration": stage_iteration,
                    "phase": phase,
                    "loss": float(loss.detach()),
                    "loss_distortion": float(loss.detach()),
                    "learning_rate": learning_rate,
                    "q_max": float(diagnostics.q_values.detach().amax()),
                    "q_p95": float(torch.quantile(diagnostics.q_values.detach(), 0.95)),
                    "minimum_extended_signed_area": float(0.5 * areas.detach().amin()),
                    "source_boundary_displacement_percent": motion[
                        "mean_displacement_percent_of_initial_radius"
                    ],
                    "source_boundary_radial_cv_percent": motion["radial_cv_percent"],
                    "is_valid": True,
                }
            )
        if stage_iteration == iterations:
            final = mapped.detach()
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip)
        if observer is not None:
            observer.gradient(stage_iteration, gradient_norm)
        optimizer.step()
    assert final is not None
    return final, history, time.perf_counter() - start


def _train_local_stage(
    module,
    base_uv,
    vertices,
    faces,
    reference_uv,
    boundary,
    *,
    iterations: int,
    learning_rate: float,
    check_interval: int,
    gradient_clip: float,
    iteration_offset: int,
    risk_adaptive: bool = False,
    risk_threshold: float = 0.85,
    risk_floor_lr: float = 1.0e-6,
    risk_global_factor: float = 0.25,
    risk_warmup_iters: int = 100,
    risk_warmup_start_factor: float = 0.25,
    dynamic_risk_threshold: float = 0.95,
    dynamic_floor_lr: float = 1.0e-8,
    risk_recovery: bool = False,
    risk_recovery_threshold: float = 0.94,
    risk_recovery_patience: int = 20,
    risk_recovery_interval: int = 10,
    risk_recovery_factor: float = 1.05,
    observer=None,
    resume_state=None,
) -> tuple[torch.Tensor, list[dict[str, Any]], float, dict[str, Any]]:
    with torch.no_grad():
        _, initial_diagnostics = module(base_uv, return_diagnostics=True)
        initial_risks = radial_conditioning_risk(
            initial_diagnostics.q_values, module.radial_map
        )
        initial_risk_max = float(initial_risks.amax())
        initial_risk_p95 = float(torch.quantile(initial_risks, 0.95))

    safe_peak_learning_rate = float(learning_rate)
    safety_triggered = bool(risk_adaptive and initial_risk_max >= risk_threshold)
    if safety_triggered:
        margin_factor = min(
            1.0,
            max(0.0, 1.0 - initial_risk_max) / (1.0 - risk_threshold),
        )
        safe_peak_learning_rate = max(
            risk_floor_lr,
            learning_rate * margin_factor * risk_global_factor,
        )
    warmup_steps = risk_warmup_iters if safety_triggered else 0
    initial_learning_rate = safe_peak_learning_rate
    if warmup_steps > 0:
        initial_learning_rate = max(
            risk_floor_lr,
            safe_peak_learning_rate * risk_warmup_start_factor,
        )
        if initial_learning_rate >= safe_peak_learning_rate:
            warmup_steps = 0

    start_iteration = 0 if resume_state is None else int(resume_state["step"])
    if not 0 <= start_iteration <= iterations:
        raise ValueError("invalid local resume step")
    if resume_state is not None:
        saved = resume_state["schedule"]
        initial_risk_max = saved["initial_risk_max"]
        initial_risk_p95 = saved["initial_risk_p95"]
        safety_triggered = saved["safety_triggered"]
        safe_peak_learning_rate = saved["safe_peak_learning_rate"]
        initial_learning_rate = saved["initial_learning_rate"]
        warmup_steps = saved["warmup_steps"]

    optimizer = torch.optim.Adam(module.parameters(), lr=initial_learning_rate)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    recovery_controller = None
    if risk_recovery:
        recovery_controller = RiskLearningRateRecovery(
            threshold=risk_recovery_threshold,
            patience=risk_recovery_patience,
            interval=risk_recovery_interval,
            factor=risk_recovery_factor,
            maximum_learning_rate=safe_peak_learning_rate,
        )
        if resume_state is not None:
            recovery_controller.load_state_dict(saved["recovery_state"])
    history: list[dict[str, Any]] = []
    learning_rate_events: list[dict[str, Any]] = []
    start = time.perf_counter()
    final = None
    for stage_iteration in range(start_iteration, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        mapped, diagnostics = module(base_uv, return_diagnostics=True)
        loss = symmetric_dirichlet_loss(
            vertices,
            faces,
            mapped[: vertices.shape[0]],
        )
        areas = signed_double_areas(mapped, module.faces)
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"non-finite loss in local stage at {stage_iteration}")
        if not bool(torch.all(areas.detach() > 0.0)):
            raise RuntimeError(
                f"hard area assertion failed in local stage at {stage_iteration}; no rollback"
            )
        risk_values = radial_conditioning_risk(
            diagnostics.q_values.detach(), module.radial_map
        )
        risk_max = float(risk_values.amax())
        risk_p95 = float(torch.quantile(risk_values, 0.95))
        # Saved states are taken after the LR controller, before the Adam update.
        already_observed = resume_state is not None and stage_iteration == start_iteration
        if not already_observed and risk_adaptive and risk_max > dynamic_risk_threshold:
            dynamic_cap = max(
                dynamic_floor_lr,
                safe_peak_learning_rate
                * min(
                    1.0,
                    max(0.0, 1.0 - risk_max)
                    / (1.0 - dynamic_risk_threshold),
                ),
            )
            current_learning_rate = float(optimizer.param_groups[0]["lr"])
            if dynamic_cap < current_learning_rate:
                for group in optimizer.param_groups:
                    group["lr"] = dynamic_cap
                learning_rate_events.append(
                    {
                        "stage_iteration": stage_iteration,
                        "reason": "dynamic_conditioning_risk",
                        "old_learning_rate": current_learning_rate,
                        "new_learning_rate": dynamic_cap,
                        "conditioning_risk_max": risk_max,
                    }
                )
        if not already_observed and recovery_controller is not None and stage_iteration >= warmup_steps:
            current_learning_rate = float(optimizer.param_groups[0]["lr"])
            recovery_event = recovery_controller.observe(
                step=stage_iteration,
                risk_max=risk_max,
                learning_rate=current_learning_rate,
            )
            if recovery_event is not None:
                for group in optimizer.param_groups:
                    group["lr"] = recovery_event.new_learning_rate
                learning_rate_events.append(
                    {
                        "stage_iteration": recovery_event.step,
                        "reason": "conditioning_risk_recovery",
                        "old_learning_rate": recovery_event.old_learning_rate,
                        "new_learning_rate": recovery_event.new_learning_rate,
                        "conditioning_risk_max": (
                            recovery_event.conditioning_risk_max
                        ),
                    }
                )
        if observer is not None:
            observer.schedule_state = {
                "initial_risk_max": initial_risk_max,
                "initial_risk_p95": initial_risk_p95,
                "safety_triggered": safety_triggered,
                "safe_peak_learning_rate": safe_peak_learning_rate,
                "initial_learning_rate": initial_learning_rate,
                "warmup_steps": warmup_steps,
                "recovery_state": recovery_controller.state_dict() if recovery_controller else None,
            }
            observer(stage_iteration, mapped, loss, diagnostics, optimizer)
        if stage_iteration % check_interval == 0 or stage_iteration == iterations:
            motion = _boundary_summary(reference_uv, mapped.detach(), boundary)
            history.append(
                {
                    "iteration": iteration_offset + stage_iteration,
                    "stage_iteration": stage_iteration,
                    "phase": (
                        f"local_{module.radial_map}_{module.latent_transform}_risk_adaptive_no_rollback"
                        if risk_adaptive
                        else f"local_{module.radial_map}_{module.latent_transform}_no_rollback"
                    ),
                    "loss": float(loss.detach()),
                    "loss_distortion": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "q_max": float(diagnostics.q_values.detach().amax()),
                    "q_p95": float(torch.quantile(diagnostics.q_values.detach(), 0.95)),
                    "conditioning_risk_max": risk_max,
                    "conditioning_risk_p95": risk_p95,
                    "minimum_extended_signed_area": float(0.5 * areas.detach().amin()),
                    "source_boundary_displacement_percent": motion[
                        "mean_displacement_percent_of_initial_radius"
                    ],
                    "source_boundary_radial_cv_percent": motion["radial_cv_percent"],
                    "is_valid": True,
                }
            )
        if stage_iteration == iterations:
            final = mapped.detach()
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip)
        if observer is not None:
            observer.gradient(stage_iteration, gradient_norm)
        optimizer.step()
        if warmup_steps > 0 and stage_iteration < warmup_steps:
            completed = stage_iteration + 1
            fraction = min(1.0, completed / warmup_steps)
            next_learning_rate = initial_learning_rate + fraction * (
                safe_peak_learning_rate - initial_learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = next_learning_rate
    assert final is not None
    learning_rate_info = {
        "risk_adaptive": risk_adaptive,
        "configured_learning_rate": learning_rate,
        "initial_risk_max": initial_risk_max,
        "initial_risk_p95": initial_risk_p95,
        "risk_threshold": risk_threshold,
        "safety_triggered": safety_triggered,
        "risk_global_factor": risk_global_factor,
        "safe_peak_learning_rate": safe_peak_learning_rate,
        "initial_learning_rate": initial_learning_rate,
        "warmup_steps": warmup_steps,
        "dynamic_risk_threshold": dynamic_risk_threshold,
        "dynamic_floor_learning_rate": dynamic_floor_lr,
        "risk_recovery": risk_recovery,
        "risk_recovery_threshold": risk_recovery_threshold,
        "risk_recovery_patience": risk_recovery_patience,
        "risk_recovery_interval": risk_recovery_interval,
        "risk_recovery_factor": risk_recovery_factor,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "events": learning_rate_events,
    }
    return final, history, time.perf_counter() - start, learning_rate_info


def _save_stage(
    output: Path,
    *,
    method: str,
    mesh,
    initial_uv: np.ndarray,
    final_extended_uv: torch.Tensor,
    extended_faces: torch.Tensor,
    source_boundary: torch.Tensor,
    history: list[dict[str, Any]],
    info: dict[str, Any],
    config: dict[str, Any],
    iterations: int,
    intersection_batch_size: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    final_uv = final_extended_uv[: len(mesh.vertices)].cpu().numpy()
    final_injectivity = validate_uv(final_uv, mesh.faces)
    extended_injectivity = validate_uv_torch(
        final_extended_uv,
        extended_faces,
        intersection_batch_size=intersection_batch_size,
    )
    if not final_injectivity["is_valid"] or not extended_injectivity["is_valid"]:
        raise RuntimeError(f"{method} failed its final injectivity audit")
    initial_injectivity = validate_uv(initial_uv, mesh.faces)
    initial_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, initial_uv)
    final_distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, final_uv)
    info["final_extended_injectivity"] = extended_injectivity
    info["boundary_motion"] = _boundary_summary(
        torch.as_tensor(initial_uv, dtype=torch.float64, device=final_extended_uv.device),
        final_extended_uv[: len(mesh.vertices)],
        source_boundary,
    )
    payload = {
        "initial": {"injectivity": initial_injectivity, "distortion": initial_distortion},
        "final": {"injectivity": final_injectivity, "distortion": final_distortion},
        "training": info,
        "history": history,
    }

    save_mesh(output, mesh, uv=final_uv)
    save_uv_plot(output.with_suffix(".uv.png"), final_uv, mesh.faces)
    save_uv_comparison_plot(
        output.with_suffix(".compare.png"),
        initial_uv,
        final_uv,
        mesh.faces,
        initial_title=(
            "Initial UV\n"
            f"SD={initial_distortion['symmetric_dirichlet_area_weighted_mean']:.4g}"
        ),
        final_title=(
            f"{method} (no rollback)\n"
            f"SD={final_distortion['symmetric_dirichlet_area_weighted_mean']:.4g}"
        ),
    )
    save_boundary_comparison(
        output.with_suffix(".boundary_compare.png"),
        initial_uv,
        final_uv,
        source_boundary.cpu().numpy(),
    )
    save_flip_heatmap(
        output.with_suffix(".flip_heatmap.png"),
        final_uv,
        mesh.faces,
        title=f"{method} Flip Heatmap",
    )
    save_distortion_comparison_heatmap(
        output.with_suffix(".distortion_compare.png"),
        mesh.vertices,
        mesh.faces,
        initial_uv,
        final_uv,
    )
    save_intersection_heatmap(
        output.with_suffix(".intersection_heatmap.png"),
        final_uv,
        mesh.faces,
        final_injectivity["intersections"],
    )
    save_loss_plot(output.with_suffix(".loss.png"), history)
    with output.with_suffix(".config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    with output.with_suffix(".metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    save_run_summary(output, build_run_summary(method, iterations, payload))
    return payload


def validate_args(args) -> None:
    if min(args.harmonic_iters, args.local_iters, args.final_harmonic_iters) < 0:
        raise ValueError("stage iteration counts must be non-negative")
    if min(args.harmonic_lr, args.local_lr) <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.check_interval < 1:
        raise ValueError("check-interval must be positive")
    for name in ("harmonic_max_log_scale", "harmonic_max_shift",
                 "final_harmonic_max_log_scale", "final_harmonic_max_shift"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")
    if args.scaffold_rings < 1:
        raise ValueError("scaffold-rings must be at least 1")
    if args.scaffold_transition_exponent <= 0.0:
        raise ValueError("scaffold-transition-exponent must be positive")
    if args.local_spline_bins < 2:
        raise ValueError("local-spline-bins must be at least 2")
    if args.local_spline_bound <= 0.0:
        raise ValueError("local-spline-bound must be positive")
    if not 0.0 <= args.boundary_hat_min_peak_weight <= 1.0:
        raise ValueError("boundary-hat-min-peak-weight must lie in [0, 1]")
    if args.local_risk_adaptive:
        for name, value in (
            ("local-risk-threshold", args.local_risk_threshold),
            ("local-dynamic-risk-threshold", args.local_dynamic_risk_threshold),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        for name, value in (
            ("local-risk-global-factor", args.local_risk_global_factor),
            ("local-risk-warmup-start-factor", args.local_risk_warmup_start_factor),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if args.local_risk_warmup_iters < 0:
            raise ValueError("local-risk-warmup-iters must be non-negative")
        if not 0.0 < args.local_risk_floor_lr <= args.local_lr:
            raise ValueError("local-risk-floor-lr must lie in (0, local-lr]")
        if not 0.0 < args.local_dynamic_floor_lr <= args.local_lr:
            raise ValueError("local-dynamic-floor-lr must lie in (0, local-lr]")
    if args.local_risk_recovery:
        if not args.local_risk_adaptive:
            raise ValueError("local-risk-recovery requires local-risk-adaptive")
        if not 0.0 < args.local_risk_recovery_threshold < 1.0:
            raise ValueError(
                "local-risk-recovery-threshold must lie strictly between 0 and 1"
            )
        if args.local_risk_recovery_threshold >= args.local_dynamic_risk_threshold:
            raise ValueError(
                "local-risk-recovery-threshold must be below "
                "local-dynamic-risk-threshold to provide hysteresis"
            )
        if min(
            args.local_risk_recovery_patience,
            args.local_risk_recovery_interval,
        ) < 1:
            raise ValueError("risk recovery patience and interval must be positive")
        if args.local_risk_recovery_factor <= 1.0:
            raise ValueError("local-risk-recovery-factor must be greater than 1")


def observed_inverse(model, uv, reference, *, include_final):
    try:
        with torch.no_grad():
            restored = model(uv, inverse=True, include_final_harmonic=include_final)
        value = float(torch.max(torch.abs(restored-reference)))
        return {"status": "ok" if np.isfinite(value) else "nonfinite",
                "legacy_max_abs_coordinate": value if np.isfinite(value) else None}
    except (torch.OutOfMemoryError, MemoryError):
        raise
    except Exception as exc:
        if any(s in str(exc).lower() for s in ("out of memory", "not enough memory", "cannot allocate memory")):
            raise
        return {"status": "failed", "legacy_max_abs_coordinate": None, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    validate_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    total_start = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    mesh = load_mesh(args.input, prim_path=args.prim_path)
    topology = validate_disk_topology(mesh.faces, len(mesh.vertices))
    uv0 = generate_initial_uv(
        mesh, method="tutte", boundary_mode="circle", geometry_scale=True
    ).uv
    vertices_cpu = torch.as_tensor(mesh.vertices, dtype=torch.float64)
    faces_cpu = torch.as_tensor(mesh.faces, dtype=torch.long)
    uv_cpu = torch.as_tensor(uv0, dtype=torch.float64)
    if args.scaffold_rings == 1:
        scaffold = build_outer_scaffold(
            vertices_cpu, faces_cpu, uv_cpu, scale=args.scaffold_scale
        )
    else:
        scaffold = build_multiring_outer_scaffold(
            vertices_cpu,
            faces_cpu,
            uv_cpu,
            scale=args.scaffold_scale,
            rings=args.scaffold_rings,
            transition_exponent=args.scaffold_transition_exponent,
        )
    transition_rings = None
    transition_weights = None
    if args.scaffold_mode_profile == "geometric":
        if args.scaffold_rings == 1:
            raise ValueError("geometric scaffold mode profile requires multiple rings")
        transition_rings, transition_weights = multiring_transition_profile(
            scaffold,
            rings=args.scaffold_rings,
            transition_exponent=args.scaffold_transition_exponent,
        )
    names, modes = build_radial_fourier_harmonic_modes(
        scaffold.uv,
        scaffold.faces,
        scaffold.original_boundary,
        scaffold.outer_boundary,
        frequencies=args.frequencies,
        transition_rings=transition_rings,
        transition_weights=transition_weights,
    )
    requested_boundary_hat_counts = tuple(args.boundary_hat_counts)
    effective_boundary_hat_counts = requested_boundary_hat_counts
    boundary_hat_support = []
    if requested_boundary_hat_counts:
        selected_counts, boundary_hat_support = select_supported_boundary_hat_counts(
            scaffold.uv,
            scaffold.original_boundary,
            requested_boundary_hat_counts,
            minimum_peak_weight=args.boundary_hat_min_peak_weight,
        )
        if args.auto_boundary_hat_counts:
            effective_boundary_hat_counts = selected_counts
            if not effective_boundary_hat_counts:
                raise ValueError("all requested boundary hat scales failed support audit")
            if effective_boundary_hat_counts != requested_boundary_hat_counts:
                print(
                    "boundary hats auto-selected "
                    f"{list(effective_boundary_hat_counts)} from "
                    f"{list(requested_boundary_hat_counts)}"
                )
    if effective_boundary_hat_counts:
        hat_names, hat_modes = build_piecewise_linear_boundary_harmonic_modes(
            scaffold.uv,
            scaffold.faces,
            scaffold.original_boundary,
            scaffold.outer_boundary,
            control_counts=effective_boundary_hat_counts,
            transition_rings=transition_rings,
            transition_weights=transition_weights,
        )
        names = names + hat_names
        modes = torch.cat((modes, hat_modes), dim=0)
    preflight = preflight_mode_intervals(
        scaffold.uv,
        scaffold.faces,
        modes,
        names,
        area_margin_ratio=args.area_margin_ratio,
    )
    model = HarmonicLocalHarmonicFlow(
        scaffold.vertices_3d,
        scaffold.faces,
        scaffold.uv,
        modes,
        names,
        harmonic_cycles=args.harmonic_cycles,
        harmonic_max_log_scale=args.harmonic_max_log_scale,
        harmonic_max_shift=args.harmonic_max_shift,
        final_harmonic_max_log_scale=args.final_harmonic_max_log_scale,
        final_harmonic_max_shift=args.final_harmonic_max_shift,
        area_margin_ratio=args.area_margin_ratio,
        local_cycles=args.local_cycles,
        local_hidden_dim=args.local_hidden_dim,
        local_radial_map=args.local_radial_map,
        local_max_log_scale=args.local_max_log_scale,
        local_max_shift_fraction=args.local_max_shift_fraction,
        local_center_iterations=args.local_center_iterations,
        local_latent_transform=args.local_latent_transform,
        local_spline_bins=args.local_spline_bins,
        local_spline_bound=args.local_spline_bound,
    ).to(device=device, dtype=torch.float64)
    vertices = vertices_cpu.to(device)
    faces = faces_cpu.to(device)
    reference = model.initial_uv
    boundary = scaffold.original_boundary.to(device)
    preprocessing_seconds = time.perf_counter() - total_start
    print(
        f"initial SD={compute_distortion_metrics(mesh.vertices, mesh.faces, uv0)['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"preflight={min(float(row['minimum_clearance']) for row in preflight):.8f}"
    )

    model.set_trainable_stage("first_harmonic")
    first_uv, first_history, first_seconds = _train_harmonic_stage(
        model.first_harmonic,
        reference,
        vertices,
        faces,
        reference,
        boundary,
        iterations=args.harmonic_iters,
        learning_rate=args.harmonic_lr,
        check_interval=args.check_interval,
        gradient_clip=args.gradient_clip,
        phase="first_global_harmonic",
        iteration_offset=0,
    )
    model.set_trainable_stage("local")
    two_stage_uv, local_history, local_seconds, local_lr_info = _train_local_stage(
        model.local,
        first_uv,
        vertices,
        faces,
        reference,
        boundary,
        iterations=args.local_iters,
        learning_rate=args.local_lr,
        check_interval=args.check_interval,
        gradient_clip=args.gradient_clip,
        iteration_offset=args.harmonic_iters,
        risk_adaptive=args.local_risk_adaptive,
        risk_threshold=args.local_risk_threshold,
        risk_floor_lr=args.local_risk_floor_lr,
        risk_global_factor=args.local_risk_global_factor,
        risk_warmup_iters=args.local_risk_warmup_iters,
        risk_warmup_start_factor=args.local_risk_warmup_start_factor,
        dynamic_risk_threshold=args.local_dynamic_risk_threshold,
        dynamic_floor_lr=args.local_dynamic_floor_lr,
        risk_recovery=args.local_risk_recovery,
        risk_recovery_threshold=args.local_risk_recovery_threshold,
        risk_recovery_patience=args.local_risk_recovery_patience,
        risk_recovery_interval=args.local_risk_recovery_interval,
        risk_recovery_factor=args.local_risk_recovery_factor,
    )
    two_inverse = observed_inverse(model, two_stage_uv, reference, include_final=False)
    two_inverse_error = two_inverse["legacy_max_abs_coordinate"]
    first_motion = _boundary_summary(reference, first_uv, boundary)
    two_motion = _boundary_summary(reference, two_stage_uv, boundary)
    output_dir = Path(args.output_dir)
    mesh_name = Path(args.input).parent.name or Path(args.input).stem
    config = dict(vars(args))
    config["requested_boundary_hat_counts"] = list(requested_boundary_hat_counts)
    config["effective_boundary_hat_counts"] = list(effective_boundary_hat_counts)
    config["boundary_hat_support_diagnostics"] = boundary_hat_support
    common_info = {
        "rollback_enabled": False,
        "hard_validity_assertions": True,
        "topology": topology,
        "preflight": preflight,
        "model": {
            "composition": (
                f"H1 -> local {args.local_latent_transform} PL-NVP -> optional H2"
            ),
            "frequencies": args.frequencies,
            "boundary_hat_counts": list(effective_boundary_hat_counts),
            "requested_boundary_hat_counts": list(requested_boundary_hat_counts),
            "boundary_hat_auto_selection": args.auto_boundary_hat_counts,
            "boundary_hat_support_diagnostics": boundary_hat_support,
            "boundary_continuity": "C0 piecewise-linear",
            "harmonic_cycles": args.harmonic_cycles,
            "local_cycles": args.local_cycles,
            "local_hidden_dim": args.local_hidden_dim,
            "local_radial_map": args.local_radial_map,
            "local_latent_transform": args.local_latent_transform,
            "local_spline_bins": args.local_spline_bins,
            "local_spline_bound": args.local_spline_bound,
            "local_spline_chart": (
                "exact_axis_interval_atanh"
                if args.local_latent_transform == "spline"
                else None
            ),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "outer_scaffold_boundary_fixed": True,
            "scaffold_rings": args.scaffold_rings,
            "scaffold_transition_exponent": args.scaffold_transition_exponent,
            "scaffold_mode_profile": args.scaffold_mode_profile,
        },
        "objective": {
            "distortion": "3d_face_area_weighted_symmetric_dirichlet",
            "faces_in_objective": "original_mesh_only",
            "jacobian_barrier": False,
            "intersection_penalty": False,
        },
        "local_learning_rate_control": local_lr_info,
    }
    two_info = {
        **common_info,
        "selected_checkpoint": "two_stage_final_no_rollback",
        "selected_iteration": args.harmonic_iters + args.local_iters,
        "inverse_max_abs_error": two_inverse_error,
        "network_inverse": two_inverse,
        "stage_boundary_motion": {
            "after_first_harmonic": first_motion,
            "after_local": two_motion,
        },
        "runtime": {
            "preprocessing_seconds": preprocessing_seconds,
            "first_harmonic_seconds": first_seconds,
            "local_seconds": local_seconds,
            "optimization_seconds": first_seconds + local_seconds,
            "total_seconds_before_artifacts": time.perf_counter() - total_start,
        },
    }
    two_output = output_dir / "two_stage" / f"{mesh_name}_harmonic_local.usda"
    two_payload = _save_stage(
        two_output,
        method="harmonic_local_pl_nvp",
        mesh=mesh,
        initial_uv=uv0,
        final_extended_uv=two_stage_uv,
        extended_faces=model.faces,
        source_boundary=boundary,
        history=first_history + local_history,
        info=two_info,
        config=config,
        iterations=args.harmonic_iters + args.local_iters,
        intersection_batch_size=args.intersection_batch_size,
    )
    torch.save(model.state_dict(), two_output.with_suffix(".model.pt"))
    print(
        f"two-stage SD={two_payload['final']['distortion']['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"boundary={two_motion['mean_displacement_percent_of_initial_radius']:.4f}% "
        f"inverse={two_inverse}"
    )

    model.set_trainable_stage("final_harmonic")
    three_stage_uv, final_history, final_seconds = _train_harmonic_stage(
        model.final_harmonic,
        two_stage_uv,
        vertices,
        faces,
        reference,
        boundary,
        iterations=args.final_harmonic_iters,
        learning_rate=args.harmonic_lr,
        check_interval=args.check_interval,
        gradient_clip=args.gradient_clip,
        phase="final_global_harmonic",
        iteration_offset=args.harmonic_iters + args.local_iters,
    )
    three_inverse = observed_inverse(model, three_stage_uv, reference, include_final=True)
    three_inverse_error = three_inverse["legacy_max_abs_coordinate"]
    with torch.no_grad():
        composed = model(include_final_harmonic=True)
    composition_error = float(torch.max(torch.abs(composed - three_stage_uv)))
    three_motion = _boundary_summary(reference, three_stage_uv, boundary)
    three_info = {
        **common_info,
        "selected_checkpoint": "three_stage_final_no_rollback",
        "selected_iteration": (
            args.harmonic_iters + args.local_iters + args.final_harmonic_iters
        ),
        "inverse_max_abs_error": three_inverse_error,
        "network_inverse": three_inverse,
        "cached_vs_composed_max_abs_error": composition_error,
        "stage_boundary_motion": {
            "after_first_harmonic": first_motion,
            "after_local": two_motion,
            "after_final_harmonic": three_motion,
        },
        "runtime": {
            "preprocessing_seconds": preprocessing_seconds,
            "first_harmonic_seconds": first_seconds,
            "local_seconds": local_seconds,
            "final_harmonic_seconds": final_seconds,
            "optimization_seconds": first_seconds + local_seconds + final_seconds,
            "total_seconds_before_artifacts": time.perf_counter() - total_start,
        },
    }
    three_output = output_dir / "three_stage" / f"{mesh_name}_harmonic_local_harmonic.usda"
    three_payload = _save_stage(
        three_output,
        method="harmonic_local_harmonic_pl_nvp",
        mesh=mesh,
        initial_uv=uv0,
        final_extended_uv=three_stage_uv,
        extended_faces=model.faces,
        source_boundary=boundary,
        history=first_history + local_history + final_history,
        info=three_info,
        config=config,
        iterations=args.harmonic_iters + args.local_iters + args.final_harmonic_iters,
        intersection_batch_size=args.intersection_batch_size,
    )
    torch.save(model.state_dict(), three_output.with_suffix(".model.pt"))
    print(
        f"three-stage SD={three_payload['final']['distortion']['symmetric_dirichlet_area_weighted_mean']:.8f} "
        f"boundary={three_motion['mean_displacement_percent_of_initial_radius']:.4f}% "
        f"inverse={three_inverse} output={output_dir}"
    )


if __name__ == "__main__":
    main()
