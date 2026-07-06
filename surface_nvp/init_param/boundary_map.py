from __future__ import annotations

import numpy as np


def _boundary_lengths(vertices: np.ndarray, loop: list[int]) -> np.ndarray:
    pts = vertices[np.asarray(loop, dtype=np.int64)]
    nxt = np.roll(pts, -1, axis=0)
    return np.linalg.norm(nxt - pts, axis=1)


def map_boundary_to_circle(vertices: np.ndarray, loop: list[int], radius: float = 1.0) -> np.ndarray:
    lengths = _boundary_lengths(vertices, loop)
    total = float(np.sum(lengths))
    if total <= 0.0:
        raise ValueError("degenerate boundary length")
    uv = np.zeros((len(loop), 2), dtype=np.float64)
    acc = 0.0
    for i, length in enumerate(lengths):
        theta = 2.0 * np.pi * acc / total
        uv[i] = [radius * np.cos(theta), radius * np.sin(theta)]
        acc += float(length)
    return uv


def map_boundary_to_square(vertices: np.ndarray, loop: list[int]) -> np.ndarray:
    lengths = _boundary_lengths(vertices, loop)
    total = float(np.sum(lengths))
    if total <= 0.0:
        raise ValueError("degenerate boundary length")
    uv = np.zeros((len(loop), 2), dtype=np.float64)
    acc = 0.0
    for i, _ in enumerate(lengths):
        t = (acc / total) % 1.0
        if t < 0.25:
            uv[i] = [4.0 * t, 0.0]
        elif t < 0.5:
            uv[i] = [1.0, 4.0 * (t - 0.25)]
        elif t < 0.75:
            uv[i] = [1.0 - 4.0 * (t - 0.5), 1.0]
        else:
            uv[i] = [0.0, 1.0 - 4.0 * (t - 0.75)]
        acc += float(lengths[i])
    return uv
