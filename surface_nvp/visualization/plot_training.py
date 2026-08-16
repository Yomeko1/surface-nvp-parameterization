from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_loss_plot(path: str | Path, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    curves = [
        ("loss", "total"),
        ("loss_distortion", "distortion"),
        ("weighted_loss_boundary", "boundary weighted"),
        ("weighted_loss_identity", "identity weighted"),
        ("weighted_loss_jacobian", "Jacobian barrier weighted"),
    ]
    valid_history = [entry for entry in history if entry.get("is_valid") and not entry.get("selected_best_valid")]
    invalid_history = [entry for entry in history if entry.get("is_valid") is False]
    max_valid_loss = 0.0
    for key, label in curves:
        points = [(entry.get("iteration"), entry.get(key)) for entry in valid_history]
        points = [(it, value) for it, value in points if it is not None and value is not None]
        if points:
            iterations, values = zip(*points)
            max_valid_loss = max(max_valid_loss, max(values))
            ax.plot(iterations, values, marker="o", linewidth=1.2, markersize=3, label=label)
    if invalid_history and max_valid_loss > 0.0:
        invalid_iterations = [entry["iteration"] for entry in invalid_history if entry.get("iteration") is not None]
        ax.scatter(invalid_iterations, [max_valid_loss] * len(invalid_iterations), marker="x", color="red", s=50, label="invalid checkpoint")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss (valid checkpoints only)")
    if ax.has_data():
        ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
