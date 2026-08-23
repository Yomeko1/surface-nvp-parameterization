from scripts.run_benchmark import (
    _benchmark_signature,
    _geometry_scale_flag,
    _global_transform_flag,
    _is_completed_record,
    _missing_summary_row,
    _output_path,
    _run_key,
    _sha256_file,
)


def test_benchmark_signature_changes_with_configuration():
    first = _benchmark_signature({"iters": 1000, "geometry_scale": True})
    second = _benchmark_signature({"iters": 2000, "geometry_scale": True})

    assert first != second
    assert _run_key("Balls", "spline", 0, first) != _run_key("Balls", "spline", 0, second)


def test_geometry_scale_flag_matches_policy():
    assert _geometry_scale_flag(True) == "--geometry-scale"
    assert _geometry_scale_flag(False) == "--no-geometry-scale"


def test_global_transform_flag_matches_effective_configuration():
    assert _global_transform_flag(True) == "--global-transform"
    assert _global_transform_flag(False) == "--no-global-transform"


def test_multi_seed_output_paths_do_not_collide(tmp_path):
    first = _output_path(tmp_path, "Balls", "spline", 0, ".obj", True)
    second = _output_path(tmp_path, "Balls", "spline", 1, ".obj", True)

    assert first != second
    assert first == tmp_path / "Balls/seed_0/spline/Balls_spline_seed0.obj"
    assert _output_path(tmp_path, "Balls", "spline", 0, ".obj", False) == (
        tmp_path / "Balls/spline/Balls_spline.obj"
    )


def test_completed_record_requires_success_path_and_metrics_digest(tmp_path):
    metrics = tmp_path / "run.metrics.json"
    metrics.write_text('{"valid": true}', encoding="utf-8")
    expected = str(metrics)
    record = {
        "returncode": 0,
        "metrics": expected,
        "metrics_sha256": _sha256_file(metrics),
        "initial_uv_sha256": "initial-a",
    }

    assert _is_completed_record(record, metrics, expected, "initial-a")
    assert not _is_completed_record(record, metrics, expected, "initial-b")
    record["returncode"] = 1
    assert not _is_completed_record(record, metrics, expected, "initial-a")
    record["returncode"] = 0
    metrics.write_text('{"valid": false}', encoding="utf-8")
    assert not _is_completed_record(record, metrics, expected, "initial-a")


def test_missing_summary_row_preserves_failed_seed(tmp_path):
    metrics = tmp_path / "missing.metrics.json"
    row = _missing_summary_row(
        "Balls", "spline", 3, 1000, {"returncode": 1, "wall_elapsed_seconds": 2.5}, metrics
    )

    assert row["dataset"] == "Balls"
    assert row["seed"] == 3
    assert row["method"] == "spline"
    assert row["status"] == "failed"
    assert row["valid"] is False
    assert row["symmetric_dirichlet_area_weighted_mean"] is None
