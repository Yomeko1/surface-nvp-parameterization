from __future__ import annotations

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection

from surface_nvp.injectivity.signed_area import triangle_signed_areas
from surface_nvp.losses.distortion import symmetric_dirichlet_per_face


def save_flip_heatmap(path: str | Path, uv: np.ndarray, faces: np.ndarray, title: str = "UV Flip Heatmap") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = triangle_signed_areas(uv, faces)
    positive = areas[areas > 0]
    danger = max(float(np.percentile(positive, 10)) * 0.25, 1e-12) if positive.size else 1e-12
    values = np.clip(areas / danger, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    collection = PolyCollection(uv[faces], array=values, cmap="RdYlGn", edgecolors="black", linewidths=0.15)
    collection.set_norm(colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{title}\nflipped={int(np.sum(areas <= 0))}, min signed area={float(np.min(areas)):.3g}"
    )
    cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("signed area / danger threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_area_comparison_heatmap(
    path: str | Path,
    initial_uv: np.ndarray,
    final_uv: np.ndarray,
    faces: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_areas = triangle_signed_areas(initial_uv, faces)
    final_areas = triangle_signed_areas(final_uv, faces)
    all_areas = np.concatenate([initial_areas, final_areas])
    vmax = max(float(np.percentile(np.abs(all_areas), 95)), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    last = None
    for ax, uv, areas, title in [
        (axes[0], initial_uv, initial_areas, "Initial signed area"),
        (axes[1], final_uv, final_areas, "Final signed area"),
    ]:
        last = PolyCollection(uv[faces], array=np.clip(areas, -vmax, vmax), cmap="RdYlGn", edgecolors="black", linewidths=0.15)
        last.set_norm(norm)
        ax.add_collection(last)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nmin={float(np.min(areas)):.3g}, flipped={int(np.sum(areas <= 0))}")
    cbar = fig.colorbar(last, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("signed UV triangle area, shared scale")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_distortion_comparison_heatmap(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    initial_uv: np.ndarray,
    final_uv: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial = _symmetric_dirichlet_np(vertices, faces, initial_uv)
    final = _symmetric_dirichlet_np(vertices, faces, final_uv)
    all_values = np.concatenate([initial, final])
    vmax = max(float(np.percentile(all_values, 95)), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    last = None
    for ax, uv, values, title in [
        (axes[0], initial_uv, initial, "Initial symmetric Dirichlet"),
        (axes[1], final_uv, final, "Final symmetric Dirichlet"),
    ]:
        last = PolyCollection(uv[faces], array=np.clip(values, 0.0, vmax), cmap="magma", edgecolors="black", linewidths=0.15)
        last.set_clim(0.0, vmax)
        ax.add_collection(last)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nmean={float(np.mean(values)):.3g}, max={float(np.max(values)):.3g}")
    cbar = fig.colorbar(last, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("per-face symmetric Dirichlet, shared clipped scale")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_distortion_heatmap(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    title: str = "Symmetric Dirichlet Distortion",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = _symmetric_dirichlet_np(vertices, faces, uv)
    vmax = max(float(np.percentile(values, 95)), 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    collection = PolyCollection(
        uv[faces],
        array=np.clip(values, 0.0, vmax),
        cmap="magma",
        edgecolors="black",
        linewidths=0.15,
    )
    collection.set_clim(0.0, vmax)
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{title}\nmean={float(np.mean(values)):.3g}, "
        f"p95={float(np.percentile(values, 95)):.3g}, max={float(np.max(values)):.3g}"
    )
    cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("per-face symmetric Dirichlet, own p95-clipped scale")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_intersection_heatmap(
    path: str | Path,
    uv: np.ndarray,
    faces: np.ndarray,
    intersections: list[tuple[int, int]] | list[list[int]],
    title: str = "UV Triangle Intersections",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(faces.shape[0], dtype=np.int64)
    if intersections:
        pairs = np.asarray(intersections, dtype=np.int64)
        np.add.at(counts, pairs[:, 0], 1)
        np.add.at(counts, pairs[:, 1], 1)
    vmax = max(int(counts.max()), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    collection = PolyCollection(
        uv[faces],
        array=counts,
        cmap="YlOrRd",
        edgecolors="black",
        linewidths=0.15,
    )
    collection.set_clim(0, vmax)
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{title}\nintersecting pairs={len(intersections)}, "
        f"involved faces={int(np.count_nonzero(counts))}"
    )
    cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("number of non-adjacent intersections involving each face")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _symmetric_dirichlet_np(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        values = symmetric_dirichlet_per_face(
            torch.as_tensor(vertices, dtype=torch.float32),
            torch.as_tensor(faces, dtype=torch.long),
            torch.as_tensor(uv, dtype=torch.float32),
        )
    return values.cpu().numpy()
