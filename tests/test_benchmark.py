from scripts.run_benchmark import (
    _benchmark_signature,
    _geometry_scale_flag,
    _is_completed_record,
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
