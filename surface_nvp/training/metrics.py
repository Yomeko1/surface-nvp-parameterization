from __future__ import annotations

import numpy as np
import torch

from surface_nvp.injectivity.validators import validate_uv, validate_uv_torch
from surface_nvp.losses.distortion import symmetric_dirichlet_per_face


def _triangle_angles(edge_lengths: np.ndarray) -> np.ndarray:
    a = edge_lengths[:, 0]
    b = edge_lengths[:, 1]
    c = edge_lengths[:, 2]
    eps = 1e-12
    angle0 = np.arccos(np.clip((a * a + b * b - c * c) / np.maximum(2.0 * a * b, eps), -1.0, 1.0))
    angle1 = np.arccos(np.clip((a * a + c * c - b * b) / np.maximum(2.0 * a * c, eps), -1.0, 1.0))
    angle2 = np.pi - angle0 - angle1
    return np.stack([angle0, angle1, angle2], axis=1)


def _edge_lengths(tri: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        ],
        axis=1,
    )


def compute_metrics(uv: np.ndarray, faces: np.ndarray, check_intersections: bool = True) -> dict:
    return validate_uv(uv, faces, check_intersections=check_intersections)


def compute_metrics_torch(uv: torch.Tensor, faces: torch.Tensor, check_intersections: bool = True, intersection_batch_size: int = 262144) -> dict:
    return validate_uv_torch(
        uv,
        faces,
        check_intersections=check_intersections,
        intersection_batch_size=intersection_batch_size,
    )


def compute_distortion_metrics(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> dict:
    v_t = torch.as_tensor(vertices, dtype=torch.float32)
    f_t = torch.as_tensor(faces, dtype=torch.long)
    uv_t = torch.as_tensor(uv, dtype=torch.float32)
    per_face = symmetric_dirichlet_per_face(v_t, f_t, uv_t).detach().cpu().numpy()

    tri3 = vertices[faces]
    len3 = _edge_lengths(tri3)
    area3 = 0.5 * np.linalg.norm(np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]), axis=1)
    tri_uv = uv[faces]
    len_uv = _edge_lengths(tri_uv)
    uv_e1 = tri_uv[:, 1] - tri_uv[:, 0]
    uv_e2 = tri_uv[:, 2] - tri_uv[:, 0]
    uv_area = 0.5 * (uv_e1[:, 0] * uv_e2[:, 1] - uv_e1[:, 1] * uv_e2[:, 0])
    abs_ratio = np.abs(uv_area) / np.maximum(area3, 1e-12)
    edge_ratio = len_uv / np.maximum(len3, 1e-12)
    scale = np.median(edge_ratio[edge_ratio > 1e-12]) if np.any(edge_ratio > 1e-12) else 1.0
    scaled_edge_ratio = edge_ratio / max(float(scale), 1e-12)
    angle_diff = np.abs(_triangle_angles(len_uv) - _triangle_angles(len3)) * 180.0 / np.pi
    area_weights = area3 / max(float(np.sum(area3)), 1e-12)
    mean_abs_uv_area = max(float(np.mean(np.abs(uv_area))), 1e-12)

    return {
        "symmetric_dirichlet_mean": float(np.mean(per_face)),
        "symmetric_dirichlet_area_weighted_mean": float(np.sum(per_face * area_weights)),
        "symmetric_dirichlet_median": float(np.median(per_face)),
        "symmetric_dirichlet_p90": float(np.percentile(per_face, 90)),
        "symmetric_dirichlet_p95": float(np.percentile(per_face, 95)),
        "symmetric_dirichlet_p99": float(np.percentile(per_face, 99)),
        "symmetric_dirichlet_max": float(np.max(per_face)),
        "uv_area_min": float(np.min(uv_area)),
        "uv_area_mean": float(np.mean(uv_area)),
        "uv_area_max": float(np.max(uv_area)),
        "normalized_uv_area_min": float(np.min(uv_area) / mean_abs_uv_area),
        "area_ratio_min": float(np.min(abs_ratio)),
        "area_ratio_mean": float(np.mean(abs_ratio)),
        "area_ratio_max": float(np.max(abs_ratio)),
        "edge_length_ratio_min": float(np.min(edge_ratio)),
        "edge_length_ratio_mean": float(np.mean(edge_ratio)),
        "edge_length_ratio_max": float(np.max(edge_ratio)),
        "scaled_edge_length_ratio_min": float(np.min(scaled_edge_ratio)),
        "scaled_edge_length_ratio_mean": float(np.mean(scaled_edge_ratio)),
        "scaled_edge_length_ratio_max": float(np.max(scaled_edge_ratio)),
        "angle_distortion_mean_deg": float(np.mean(angle_diff)),
        "angle_distortion_p95_deg": float(np.percentile(angle_diff, 95)),
        "angle_distortion_max_deg": float(np.max(angle_diff)),
    }
