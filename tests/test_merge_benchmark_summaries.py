import json

import pytest

from scripts.merge_benchmark_summaries import merge_benchmark_summaries


def _write_benchmark(root, dataset, source_hash="source-a"):
    root.mkdir()
    manifest = {
        "source_sha256": source_hash,
        "methods": ["direct", "affine", "spline", "slim"],
        "seed": 0,
        "datasets": {dataset: {"path": f"{dataset}.obj", "sha256": "input-a"}},
    }
    rows = [
        {"dataset": dataset, "seed": 0, "method": method, "valid": True}
        for method in ("initial", "direct", "affine", "spline", "slim")
    ]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(rows), encoding="utf-8")


def test_merge_benchmark_summaries_writes_complete_provenance(tmp_path):
    balls = tmp_path / "balls"
    cow = tmp_path / "cow"
    output = tmp_path / "all"
    _write_benchmark(balls, "Balls")
    _write_benchmark(cow, "Cow")

    rows = merge_benchmark_summaries([cow, balls], output)

    assert [(row["dataset"], row["method"]) for row in rows[:2]] == [
        ("Balls", "initial"),
        ("Balls", "direct"),
    ]
    merged_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert merged_manifest["row_count"] == 10
    assert merged_manifest["method_row_count"] == 8
    assert merged_manifest["failed_method_row_count"] == 0
    assert len(merged_manifest["components"]) == 2
    assert (output / "summary.csv").is_file()


def test_merge_benchmark_summaries_rejects_incompatible_source(tmp_path):
    balls = tmp_path / "balls"
    cow = tmp_path / "cow"
    _write_benchmark(balls, "Balls")
    _write_benchmark(cow, "Cow", source_hash="source-b")

    with pytest.raises(ValueError, match="incompatible source_sha256"):
        merge_benchmark_summaries([balls, cow], tmp_path / "all")
