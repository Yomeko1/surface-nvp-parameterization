from __future__ import annotations

import json
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "init": {
        "method": "tutte",
        "boundary": "circle",
        "geometry_scale": True,
        "initial_uv": None,
        "abfpp_executable": None,
    },
    "model": {
        "coupling_type": "affine",
        "num_layers": 6,
        "hidden_dim": 64,
        "mlp_layers": 3,
        "s_clamp": 2.0,
        "spline_bins": 16,
        "spline_bound": 1.1,
        "global_transform": True,
        "mixing_type": "none",
    },
    "train": {
        "seed": 0,
        "iters": 1000,
        "lr": 1e-3,
        "check_interval": 25,
        "plateau_patience": 4,
        "lr_decay": 0.5,
        "min_lr": 1e-6,
        "lr_schedule": "constant",
        "lbfgs_iters": 0,
        "lbfgs_lr": 1.0,
        "lbfgs_check_interval": 1,
        "boundary_weight": 0.0,
        "identity_weight": 1e-3,
        "area_weight": 100.0,
        "device": "cpu",
        "validation_device": None,
        "intersection_batch_size": 262144,
    },
    "io": {"prim_path": None},
}

COMMON_OVERRIDE_PATHS = {
    "method": ("init", "method"),
    "boundary": ("init", "boundary"),
    "initial_uv": ("init", "initial_uv"),
    "abfpp_executable": ("init", "abfpp_executable"),
    "geometry_scale": ("init", "geometry_scale"),
    "seed": ("train", "seed"),
    "iters": ("train", "iters"),
    "lr": ("train", "lr"),
    "check_interval": ("train", "check_interval"),
    "plateau_patience": ("train", "plateau_patience"),
    "lr_decay": ("train", "lr_decay"),
    "min_lr": ("train", "min_lr"),
    "lr_schedule": ("train", "lr_schedule"),
    "lbfgs_iters": ("train", "lbfgs_iters"),
    "lbfgs_lr": ("train", "lbfgs_lr"),
    "lbfgs_check_interval": ("train", "lbfgs_check_interval"),
    "boundary_weight": ("train", "boundary_weight"),
    "identity_weight": ("train", "identity_weight"),
    "area_weight": ("train", "area_weight"),
    "device": ("train", "device"),
    "validation_device": ("train", "validation_device"),
    "intersection_batch_size": ("train", "intersection_batch_size"),
    "prim_path": ("io", "prim_path"),
}

NVP_OVERRIDE_PATHS = {
    **COMMON_OVERRIDE_PATHS,
    "coupling_type": ("model", "coupling_type"),
    "num_layers": ("model", "num_layers"),
    "hidden_dim": ("model", "hidden_dim"),
    "mlp_layers": ("model", "mlp_layers"),
    "s_clamp": ("model", "s_clamp"),
    "spline_bins": ("model", "spline_bins"),
    "spline_bound": ("model", "spline_bound"),
    "global_transform": ("model", "global_transform"),
    "mixing_type": ("model", "mixing_type"),
}


def load_config(path: str | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a YAML mapping")
    return _deep_update(config, loaded)


def apply_cli_overrides(config: dict, args, paths: dict[str, tuple[str, str]]) -> dict:
    for attr, path in paths.items():
        value = getattr(args, attr, None)
        if value is not None:
            config[path[0]][path[1]] = value
    return config


def _deep_update(base: dict, override: dict, prefix: str = "") -> dict:
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"unknown config key: {path}")
        expected = base[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ValueError(f"config key must be a mapping: {path}")
            _deep_update(base[key], value, path)
        else:
            _validate_config_value(path, expected, value)
            base[key] = value
    return base


def _validate_config_value(path: str, expected, value) -> None:
    if expected is None:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"config key must be a string or null: {path}")
        return
    if isinstance(expected, bool):
        valid = isinstance(value, bool)
    elif isinstance(expected, int):
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(expected, float):
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, type(expected))
    if not valid:
        raise ValueError(f"invalid value type for config key: {path}")
