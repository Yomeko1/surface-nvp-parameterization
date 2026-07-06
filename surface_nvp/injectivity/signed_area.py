from __future__ import annotations

import numpy as np
import torch


def triangle_signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = uv[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def count_flipped(uv: np.ndarray, faces: np.ndarray, eps: float = 0.0) -> int:
    return int(np.sum(triangle_signed_areas(uv, faces) <= eps))


def torch_signed_areas(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = uv[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
