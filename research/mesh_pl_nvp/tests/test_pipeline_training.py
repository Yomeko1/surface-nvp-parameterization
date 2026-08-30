import numpy as np

from research.mesh_pl_nvp.mesh_coupling import make_grid_triangulation
from research.mesh_pl_nvp.pipeline_config import load_pipeline_config
from research.mesh_pl_nvp.pipeline_training import (
    train_mesh_pl_nvp,
    validate_disk_topology,
)


def _small_surface():
    uv, faces = make_grid_triangulation(3, 3)
    x, y = uv[:, 0], uv[:, 1]
    vertices = np.column_stack((x.numpy(), y.numpy(), (0.15 * x * y).numpy()))
    return vertices, faces.numpy(), uv.numpy()


def test_pipeline_trainer_preserves_hard_validity_without_rollback() -> None:
    vertices, faces, uv = _small_surface()
    config = load_pipeline_config(None)
    config["model"]["cycles"] = 1
    config["model"]["hidden_dim"] = 8
    config["train"]["iters"] = 2
    config["train"]["check_interval"] = 1
    config["train"]["lr_schedule"] = "constant"
    config["scaffold"]["enabled"] = True

    result = train_mesh_pl_nvp(vertices, faces, uv, config)

    assert result.info["rollback_enabled"] is False
    assert result.info["selected_iteration"] == 2
    assert result.info["scaffold"]["loss_includes_scaffold_faces"] is False
    assert result.info["q"]["released_source_boundary_updates"] is not None
    assert result.info["inverse_max_abs_error"] < 1.0e-8
    assert len(result.history) == 3


def test_topology_check_reports_disk_counts() -> None:
    vertices, faces, _ = _small_surface()
    topology = validate_disk_topology(faces, len(vertices))
    assert topology["euler_characteristic"] == 1
    assert topology["boundary_vertex_count"] == 8
