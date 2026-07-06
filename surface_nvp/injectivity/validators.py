from __future__ import annotations

import numpy as np
import torch

from .signed_area import count_flipped, triangle_signed_areas
from .signed_area import torch_signed_areas
from .triangle_intersection import count_triangle_intersections_torch, find_triangle_intersections


def validate_uv(uv: np.ndarray, faces: np.ndarray, area_eps: float = 1e-12, check_intersections: bool = True) -> dict:
    areas = triangle_signed_areas(uv, faces)
    flipped = count_flipped(uv, faces, eps=area_eps)
    intersections = []
    if check_intersections:
        intersections = find_triangle_intersections(uv, faces)
    return {
        "num_flipped": int(flipped),
        "min_signed_area": float(np.min(areas)) if areas.size else 0.0,
        "num_intersections": int(len(intersections)),
        "intersections": intersections,
        "is_valid": flipped == 0 and len(intersections) == 0,
    }


def validate_uv_torch(
    uv: torch.Tensor,
    faces: torch.Tensor,
    area_eps: float = 1e-12,
    check_intersections: bool = True,
    intersection_batch_size: int = 262144,
) -> dict:
    with torch.no_grad():
        areas = torch_signed_areas(uv, faces)
        flipped = int((areas <= area_eps).sum().item())
        num_intersections = 0
        if check_intersections:
            num_intersections = count_triangle_intersections_torch(
                uv,
                faces,
                batch_size=intersection_batch_size,
            )
        return {
            "num_flipped": flipped,
            "min_signed_area": float(areas.min().item()) if areas.numel() else 0.0,
            "num_intersections": int(num_intersections),
            "intersections": [],
            "is_valid": flipped == 0 and num_intersections == 0,
        }
