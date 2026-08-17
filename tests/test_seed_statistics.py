import pytest

from scripts.summarize_benchmark_seeds import summarize_seed_rows, validate_seed_rows


def _row(seed, method, value, valid=True):
    return {
        "dataset": "Balls",
        "seed": seed,
        "method": method,
        "status": "complete" if valid else "failed",
        "valid": valid,
        "symmetric_dirichlet_area_weighted_mean": value,
        "symmetric_dirichlet_p95": value + 1,
        "symmetric_dirichlet_p99": value + 2,
        "angle_distortion_mean_deg": value + 3,
        "angle_distortion_p95_deg": value + 4,
        "min_signed_area": value / 10,
        "selected_iteration": 100,
        "wall_elapsed_seconds": 10,
        "training_elapsed_seconds": 5,
    }


def test_seed_statistics_include_sample_spread_and_paired_wins():
    rows = [
        _row(0, "affine", 10),
        _row(1, "affine", 12),
        _row(0, "spline", 8),
        _row(1, "spline", 13, valid=False),
    ]

    statistics_rows, paired_rows = summarize_seed_rows(rows)

    spline = next(row for row in statistics_rows if row["method"] == "spline")
    assert spline["valid_rate"] == 0.5
    assert spline["symmetric_dirichlet_area_weighted_mean_mean"] == 8
    assert spline["symmetric_dirichlet_area_weighted_mean_std"] is None
    assert paired_rows[0]["pair_count"] == 1
    assert paired_rows[0]["candidate_win_rate"] == 1
    assert paired_rows[0]["relative_improvement_mean"] == pytest.approx(0.2)


def test_seed_summary_matrix_must_match_manifest():
    manifest = {
        "datasets": {"Balls": {}},
        "methods": ["affine", "spline"],
        "seeds": [0, 1],
    }
    rows = [
        _row(0, "affine", 10),
        _row(1, "affine", 12),
        _row(0, "spline", 8),
    ]

    with pytest.raises(ValueError, match="does not match manifest"):
        validate_seed_rows(rows, manifest)
