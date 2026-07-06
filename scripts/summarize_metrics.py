from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.training.summary import SUMMARY_FIELDS, build_run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="metrics.json files")
    parser.add_argument("--output", required=True, help="summary CSV path")
    parser.add_argument("--method", default=None, help="override method name for all inputs")
    args = parser.parse_args()

    rows = []
    for input_path in args.inputs:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        method = args.method or _infer_method(path)
        iters = _infer_iters(payload, path)
        row = build_run_summary(method, iters, payload)
        row["source"] = str(path)
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", *SUMMARY_FIELDS]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    with output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _infer_method(path: Path) -> str:
    text = str(path).lower()
    if "direct" in text:
        return "direct_uv"
    if "nvp" in text:
        return "nvp"
    return "unknown"


def _infer_iters(payload: dict, path: Path) -> int:
    history = payload.get("history") or []
    if history:
        return int(max(entry.get("iteration", 0) for entry in history))
    for part in path.parts:
        if part.isdigit():
            return int(part)
        if "_" in part:
            tail = part.rsplit("_", 1)[-1]
            if tail.isdigit():
                return int(tail)
    return 0


if __name__ == "__main__":
    main()
