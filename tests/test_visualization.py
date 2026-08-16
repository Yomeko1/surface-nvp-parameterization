import numpy as np

from surface_nvp.visualization.uv_diagnostics import save_distortion_heatmap, save_intersection_heatmap


def test_individual_distortion_and_intersection_heatmaps_are_written(tmp_path):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = vertices[:, :2].copy()
    distortion_path = tmp_path / "distortion.png"
    intersection_path = tmp_path / "intersection.png"

    save_distortion_heatmap(distortion_path, vertices, faces, uv)
    save_intersection_heatmap(intersection_path, uv, faces, [(0, 1)])

    assert distortion_path.stat().st_size > 0
    assert intersection_path.stat().st_size > 0
