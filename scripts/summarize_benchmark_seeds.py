from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "symmetric_dirichlet_area_weighted_mean",
    "symmetric_dirichlet_p95",
    "symmetric_dirichlet_p99",
    "angle_distortion_mean_deg",
    "angle_distortion_p95_deg",
    "min_signed_area",
    "selected_iteration",
    "wall_elapsed_seconds",
    "training_elapsed_seconds",
)
STATISTICS_FIELDS = [
    "dataset",
    "method",
    "seed_count",
    "complete_count",
    "valid_count",
    "valid_rate",
    *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std", "median", "min", "max")],
]
PAIRED_FIELDS = [
    "dataset",
    "baseline_method",
    "candidate_method",
    "pair_count",
    "candidate_win_count",
    "candidate_win_rate",
    "area_weighted_sd_delta_mean",
    "area_weighted_sd_delta_std",
    "relative_improvement_mean",
    "relative_improvement_std",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a multi-seed benchmark")
    parser.add_argument("--input", required=True, help="benchmark summary.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline", default="affine")
    parser.add_argument("--candidate", default="spline")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_root = _resolve_path(args.output_root)
    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    with (input_path.parent / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_seed_rows(rows, manifest)
    statistics_rows, paired_rows = summarize_seed_rows(rows, args.baseline, args.candidate)
    write_seed_statistics(output_root, statistics_rows, paired_rows)


def summarize_seed_rows(
    rows: list[dict], baseline_method: str = "affine", candidate_method: str = "spline"
) -> tuple[list[dict], list[dict]]:
    method_rows = [row for row in rows if row.get("method") != "initial"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in method_rows:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)

    statistics_rows = []
    for (dataset, method), group in sorted(grouped.items()):
        successful = [
            row for row in group if row.get("status") == "complete" and row.get("valid")
        ]
        result = {
            "dataset": dataset,
            "method": method,
            "seed_count": len(group),
            "complete_count": sum(row.get("status") == "complete" for row in group),
            "valid_count": len(successful),
            "valid_rate": len(successful) / len(group),
        }
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in successful
                if row.get(metric) is not None
            ]
            result.update(_describe(metric, values))
        statistics_rows.append(result)

    by_run = {(row["dataset"], int(row["seed"]), row["method"]): row for row in method_rows}
    datasets = sorted({row["dataset"] for row in method_rows})
    paired_rows = []
    metric = "symmetric_dirichlet_area_weighted_mean"
    for dataset in datasets:
        seeds = sorted(
            {
                seed
                for name, seed, method in by_run
                if name == dataset and method in {baseline_method, candidate_method}
            }
        )
        pairs = [
            (by_run.get((dataset, seed, baseline_method)), by_run.get((dataset, seed, candidate_method)))
            for seed in seeds
        ]
        pairs = [
            pair
            for pair in pairs
            if pair[0] is not None
            and pair[1] is not None
            and pair[0].get("status") == "complete"
            and pair[1].get("status") == "complete"
            and pair[0].get("valid")
            and pair[1].get("valid")
        ]
        if not pairs:
            continue
        deltas = [float(candidate[metric]) - float(baseline[metric]) for baseline, candidate in pairs]
        improvements = [
            (float(baseline[metric]) - float(candidate[metric])) / float(baseline[metric])
            for baseline, candidate in pairs
        ]
        paired_rows.append(
            {
                "dataset": dataset,
                "baseline_method": baseline_method,
                "candidate_method": candidate_method,
                "pair_count": len(pairs),
                "candidate_win_count": sum(delta < 0 for delta in deltas),
                "candidate_win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
                "area_weighted_sd_delta_mean": statistics.fmean(deltas),
                "area_weighted_sd_delta_std": _sample_std(deltas),
                "relative_improvement_mean": statistics.fmean(improvements),
                "relative_improvement_std": _sample_std(improvements),
            }
        )
    return statistics_rows, paired_rows


def validate_seed_rows(rows: list[dict], manifest: dict) -> None:
    if "seeds" not in manifest:
        raise ValueError("manifest is not a multi-seed benchmark")
    expected = {
        (dataset, int(seed), method)
        for dataset in manifest["datasets"]
        for seed in manifest["seeds"]
        for method in manifest["methods"]
    }
    method_rows = [row for row in rows if row.get("method") != "initial"]
    actual = {(row["dataset"], int(row["seed"]), row["method"]) for row in method_rows}
    if len(actual) != len(method_rows):
        raise ValueError("summary contains duplicate dataset/seed/method rows")
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"summary matrix does not match manifest; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def write_seed_statistics(output_root: Path, statistics_rows: list[dict], paired_rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "statistics.json", statistics_rows)
    _write_csv(output_root / "statistics.csv", STATISTICS_FIELDS, statistics_rows)
    _write_json(output_root / "paired.json", paired_rows)
    _write_csv(output_root / "paired.csv", PAIRED_FIELDS, paired_rows)


def _describe(metric: str, values: list[float]) -> dict:
    if not values:
        return {f"{metric}_{stat}": None for stat in ("mean", "std", "median", "min", "max")}
    return {
        f"{metric}_mean": statistics.fmean(values),
        f"{metric}_std": _sample_std(values),
        f"{metric}_median": statistics.median(values),
        f"{metric}_min": min(values),
        f"{metric}_max": max(values),
    }


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


if __name__ == "__main__":
    main()
