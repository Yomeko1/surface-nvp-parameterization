from __future__ import annotations

import csv
import json
from pathlib import Path


SUMMARY_FIELDS = [
    "method",
    "iters",
    "selected_iteration",
    "valid",
    "num_flipped",
    "num_nonfinite",
    "num_intersections",
    "min_signed_area",
    "symmetric_dirichlet_mean",
    "symmetric_dirichlet_area_weighted_mean",
    "symmetric_dirichlet_p95",
    "symmetric_dirichlet_p99",
    "symmetric_dirichlet_max",
    "angle_distortion_mean_deg",
    "angle_distortion_p95_deg",
    "angle_distortion_max_deg",
    "normalized_uv_area_min",
    "initial_symmetric_dirichlet_mean",
    "initial_symmetric_dirichlet_max",
]


def build_run_summary(method: str, iters: int, metrics_payload: dict) -> dict:
    initial = metrics_payload["initial"]
    final = metrics_payload["final"]
    selected_iteration = None
    training = metrics_payload.get("training") or {}
    if training.get("selected_iteration") is not None:
        selected_iteration = training["selected_iteration"]
    else:
        selected = _selected_history_entry(metrics_payload.get("history", []))
        selected_iteration = selected.get("iteration") if selected else None
    return {
        "method": method,
        "iters": int(iters),
        "selected_iteration": selected_iteration,
        "valid": bool(final["injectivity"]["is_valid"]),
        "num_flipped": int(final["injectivity"]["num_flipped"]),
        "num_nonfinite": int(final["injectivity"].get("num_nonfinite", 0)),
        "num_intersections": int(final["injectivity"]["num_intersections"]),
        "min_signed_area": float(final["injectivity"]["min_signed_area"]),
        "symmetric_dirichlet_mean": float(final["distortion"]["symmetric_dirichlet_mean"]),
        "symmetric_dirichlet_area_weighted_mean": _optional_float(final["distortion"], "symmetric_dirichlet_area_weighted_mean"),
        "symmetric_dirichlet_p95": _optional_float(final["distortion"], "symmetric_dirichlet_p95"),
        "symmetric_dirichlet_p99": _optional_float(final["distortion"], "symmetric_dirichlet_p99"),
        "symmetric_dirichlet_max": float(final["distortion"]["symmetric_dirichlet_max"]),
        "angle_distortion_mean_deg": float(final["distortion"]["angle_distortion_mean_deg"]),
        "angle_distortion_p95_deg": _optional_float(final["distortion"], "angle_distortion_p95_deg"),
        "angle_distortion_max_deg": float(final["distortion"]["angle_distortion_max_deg"]),
        "normalized_uv_area_min": _optional_float(final["distortion"], "normalized_uv_area_min"),
        "initial_symmetric_dirichlet_mean": float(initial["distortion"]["symmetric_dirichlet_mean"]),
        "initial_symmetric_dirichlet_max": float(initial["distortion"]["symmetric_dirichlet_max"]),
    }


def _optional_float(values: dict, key: str) -> float | None:
    value = values.get(key)
    return float(value) if value is not None else None


def save_run_summary(path_prefix: str | Path, summary: dict) -> None:
    path_prefix = Path(path_prefix)
    json_path = path_prefix.with_suffix(".summary.json")
    csv_path = path_prefix.with_suffix(".summary.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({field: summary.get(field) for field in SUMMARY_FIELDS})


def _selected_history_entry(history: list[dict]) -> dict | None:
    for entry in reversed(history):
        if entry.get("selected_best_valid"):
            return entry
    return None
