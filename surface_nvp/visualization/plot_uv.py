from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_uv_plot(path: str | Path, uv: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    _plot_uv_axes(ax, uv, faces)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_uv_comparison_plot(
    path: str | Path,
    initial_uv: np.ndarray,
    final_uv: np.ndarray,
    faces: np.ndarray,
    initial_title: str = "Initial UV",
    final_title: str = "Final UV",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _plot_uv_axes(axes[0], initial_uv, faces)
    _plot_uv_axes(axes[1], final_uv, faces)
    axes[0].set_title(initial_title)
    axes[1].set_title(final_title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_uv_axes(ax, uv: np.ndarray, faces: np.ndarray) -> None:
    for face in faces:
        pts = uv[face]
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=0.4)
    ax.set_aspect("equal", adjustable="box")
