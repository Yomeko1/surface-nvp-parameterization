"""Strict configuration handling for the isolated mesh PL-NVP runner."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "init": {
        "boundary": "circle",
        "geometry_scale": True,
        "initial_uv": None,
    },
    "model": {
        "cycles": 4,
        "hidden_dim": 32,
        "conditioner_features": "basic",
        "max_log_scale": 0.08,
        "max_shift_fraction": 0.04,
        "center_iterations": 12,
    },
    "train": {
        "seed": 20260830,
        "iters": 1000,
        "lr": 0.003,
        "device": "cpu",
        "check_interval": 10,
        "gradient_clip": 10.0,
        "lr_schedule": "adaptive-plateau",
        "min_lr": 0.0001,
        "plateau_window": 100,
        "plateau_patience": 2,
        "plateau_relative_threshold": 0.008,
        "plateau_factor": 0.5,
        "plateau_q_threshold": 0.97,
        "plateau_minimum_area_ratio": 0.25,
        "intersection_batch_size": 262144,
    },
    "scaffold": {
        "enabled": True,
        "scale": 1.1,
    },
    "io": {
        "prim_path": None,
        "save_model": True,
    },
}


CLI_OVERRIDE_PATHS = {
    "boundary": ("init", "boundary"),
    "geometry_scale": ("init", "geometry_scale"),
    "initial_uv": ("init", "initial_uv"),
    "cycles": ("model", "cycles"),
    "hidden_dim": ("model", "hidden_dim"),
    "conditioner_features": ("model", "conditioner_features"),
    "max_log_scale": ("model", "max_log_scale"),
    "max_shift_fraction": ("model", "max_shift_fraction"),
    "center_iterations": ("model", "center_iterations"),
    "seed": ("train", "seed"),
    "iters": ("train", "iters"),
    "lr": ("train", "lr"),
    "device": ("train", "device"),
    "check_interval": ("train", "check_interval"),
    "gradient_clip": ("train", "gradient_clip"),
    "lr_schedule": ("train", "lr_schedule"),
    "min_lr": ("train", "min_lr"),
    "plateau_window": ("train", "plateau_window"),
    "plateau_patience": ("train", "plateau_patience"),
    "plateau_relative_threshold": ("train", "plateau_relative_threshold"),
    "plateau_factor": ("train", "plateau_factor"),
    "plateau_q_threshold": ("train", "plateau_q_threshold"),
    "plateau_minimum_area_ratio": ("train", "plateau_minimum_area_ratio"),
    "intersection_batch_size": ("train", "intersection_batch_size"),
    "scaffold": ("scaffold", "enabled"),
    "scaffold_scale": ("scaffold", "scale"),
    "prim_path": ("io", "prim_path"),
    "save_model": ("io", "save_model"),
}


def load_pipeline_config(path: str | Path | None) -> dict[str, Any]:
    """Load a research config without extending the v2.4 production schema."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a YAML mapping")
    _deep_update(config, loaded)
    validate_pipeline_config(config)
    return config


def apply_cli_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for attribute, (section, key) in CLI_OVERRIDE_PATHS.items():
        value = getattr(args, attribute, None)
        if value is not None:
            result[section][key] = value
    validate_pipeline_config(result)
    return result


def validate_pipeline_config(config: dict[str, Any]) -> None:
    model = config["model"]
    train = config["train"]
    scaffold = config["scaffold"]
    if model["cycles"] <= 0 or model["hidden_dim"] <= 0 or model["center_iterations"] <= 0:
        raise ValueError("cycles, hidden_dim, and center_iterations must be positive")
    if model["conditioner_features"] not in {"basic", "local-geometry"}:
        raise ValueError("conditioner_features must be 'basic' or 'local-geometry'")
    if min(model["max_log_scale"], model["max_shift_fraction"]) < 0.0:
        raise ValueError("coupling bounds must be non-negative")
    if train["iters"] < 0 or train["check_interval"] <= 0:
        raise ValueError("iters must be non-negative and check_interval must be positive")
    if train["lr"] <= 0.0 or not 0.0 < train["min_lr"] <= train["lr"]:
        raise ValueError("learning rates must satisfy 0 < min_lr <= lr")
    if train["gradient_clip"] <= 0.0 or train["intersection_batch_size"] <= 0:
        raise ValueError("gradient_clip and intersection_batch_size must be positive")
    if train["lr_schedule"] not in {"constant", "adaptive-plateau"}:
        raise ValueError("lr_schedule must be 'constant' or 'adaptive-plateau'")
    if scaffold["enabled"] and scaffold["scale"] <= 1.0:
        raise ValueError("scaffold scale must be greater than one")


def _deep_update(base: dict[str, Any], override: dict[str, Any], prefix: str = "") -> None:
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"unknown config key: {path}")
        expected = base[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ValueError(f"config key must be a mapping: {path}")
            _deep_update(expected, value, path)
            continue
        if expected is None:
            if value is not None and not isinstance(value, str):
                raise ValueError(f"config key must be a string or null: {path}")
        elif isinstance(expected, bool):
            if not isinstance(value, bool):
                raise ValueError(f"config key must be boolean: {path}")
        elif isinstance(expected, int):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"config key must be integer: {path}")
        elif isinstance(expected, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"config key must be numeric: {path}")
        elif not isinstance(value, type(expected)):
            raise ValueError(f"invalid value type for config key: {path}")
        base[key] = value
