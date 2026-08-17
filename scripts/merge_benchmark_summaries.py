from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from surface_nvp.training.summary import SUMMARY_FIELDS


DATASET_ORDER = ("Balls", "David328", "NefertitiFace", "Cow", "Isis")
METHOD_ORDER = ("initial", "direct", "affine", "spline", "slim")
COMPATIBILITY_KEYS = (
    "source_sha256",
    "python",
    "platform",
    "packages",
    "torch_cuda",
    "cudnn",
    "gpu",
    "methods",
    "seed",
    "iters",
    "slim_iters",
    "check_interval",
    "device",
    "validation_device",
    "intersection_batch_size",
    "config_sha256",
    "geometry_scale",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compatible benchmark summaries")
    parser.add_argument("--inputs", nargs="+", required=True, help="benchmark output roots")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    merge_benchmark_summaries([Path(path) for path in args.inputs], Path(args.output_root))


def merge_benchmark_summaries(input_roots: list[Path], output_root: Path) -> list[dict]:
    if not input_roots:
        raise ValueError("at least one benchmark root is required")

    manifests = [_read_json(root / "manifest.json") for root in input_roots]
    reference = manifests[0]
    for root, manifest in zip(input_roots[1:], manifests[1:]):
        for key in COMPATIBILITY_KEYS:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"incompatible {key} in {root}")

    rows: list[dict] = []
    datasets: dict[str, dict] = {}
    seen_runs: set[tuple[str, int, str]] = set()
    components = []
    for root, manifest in zip(input_roots, manifests):
        overlap = datasets.keys() & manifest["datasets"].keys()
        if overlap:
            raise ValueError(f"duplicate datasets: {', '.join(sorted(overlap))}")
        datasets.update(manifest["datasets"])

        summary_path = root / "summary.json"
        summary = _read_json(summary_path)
        declared_datasets = set(manifest["datasets"])
        for row in summary:
            if row.get("dataset") not in declared_datasets:
                raise ValueError(f"summary dataset is not declared by {root}: {row.get('dataset')}")
            key = (row["dataset"], int(row["seed"]), row["method"])
            if key in seen_runs:
                raise ValueError(f"duplicate summary row: {key}")
            seen_runs.add(key)
            rows.append(row)

        components.append(
            {
                "root": str(root),
                "git_commit": manifest.get("git_commit"),
                "git_status": manifest.get("git_status"),
                "run_signature": manifest.get("run_signature"),
                "slim_sha256": manifest.get("slim_sha256"),
                "manifest_sha256": _sha256_file(root / "manifest.json"),
                "summary_sha256": _sha256_file(summary_path),
            }
        )

    expected_methods = set(reference["methods"]) | {"initial"}
    for dataset in datasets:
        actual_methods = {method for name, _, method in seen_runs if name == dataset}
        if actual_methods != expected_methods:
            raise ValueError(f"incomplete method set for {dataset}: {sorted(actual_methods)}")

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    method_rank = {name: index for index, name in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            dataset_rank.get(row["dataset"], len(dataset_rank)),
            row["dataset"],
            int(row["seed"]),
            method_rank.get(row["method"], len(method_rank)),
            row["method"],
        )
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "summary.json", rows)
    fields = [
        "dataset",
        "seed",
        "status",
        "wall_elapsed_seconds",
        "training_elapsed_seconds",
        "source",
        *SUMMARY_FIELDS,
    ]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    method_rows = [row for row in rows if row["method"] != "initial"]
    merged_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "compatibility": {key: reference.get(key) for key in COMPATIBILITY_KEYS},
        "datasets": datasets,
        "row_count": len(rows),
        "method_row_count": len(method_rows),
        "valid_method_row_count": sum(bool(row.get("valid")) for row in method_rows),
        "failed_method_row_count": sum(not bool(row.get("valid")) for row in method_rows),
    }
    _write_json(output_root / "manifest.json", merged_manifest)
    return rows


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
