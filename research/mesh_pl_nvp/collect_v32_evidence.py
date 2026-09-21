"""Collect compact release tables without copying raw checkpoints into Git."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .run_v31_audit import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(workspace, replay, output):
    output.mkdir(parents=True, exist_ok=False)
    root = workspace/"data/output"
    cases = {
        "Cow": [("v3.1", root/"v3_1_cow_validation/20260920_full_01/original_affine", 100),
                ("original_h2_300", root/"v3_1_cow_validation/20260920_full_01/original_affine", 300),
                ("v3.2", root/"v3_1_cow_validation/20260920_full_01/h2_affine_wide", 300)],
        "00027": [("v3.1", root/"v3_1_audit_00027/20260917_full_01", 100),
                  ("original_h2_300", root/"v3_1_followup_00027/20260920_full_01/h2_extend", 300),
                  ("v3.2", root/"v3_1_followup_00027/20260920_full_01/h2_affine_wide", 300)]}
    results = []
    for mesh, configurations in cases.items():
        for label, directory, step in configurations:
            path = directory/"audit"/f"final_harmonic_{step:04d}.json"
            row = read(path)
            manifest = read(directory/"manifest.json")
            results.append({"mesh": mesh, "configuration": label, "h2_updates": step,
                "seed": manifest["config"]["seed"], "input_sha256": manifest["input_sha256"],
                "source_audit": path.relative_to(workspace).as_posix(),
                "source_audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sd": row["sd"], "geometry": row["geometry"],
                "geometry_export_decode": row["geometry_export_decode"],
                "original_validity": row["original_validity"],
                "extended_validity": row["extended_validity"], "export_validity": row["export_validity"],
                "network": {kind: {key: value for key, value in row[f"network_{kind}"].items()
                                   if key != "traceback"} for kind in ("composed", "isolated")}})
    write_json(output/"v3_2_results.json", {"schema_version": 1, "release": "v3.2",
        "experiment_date": "2026-09-20", "collected_date": "2026-09-21",
        "scope": "two meshes, one seed, fixed budgets; not a universal performance guarantee",
        "export_format": "usda", "error_thresholds": None, "rows": results})
    write_json(output/"v3_2_replay.json", {"date": "2026-09-21", "device": "cuda",
        "scope": "final-state forward and selected training segments, not a second complete training run",
        "results": read(replay/"verification.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.workspace.resolve(), args.replay.resolve(), args.output.resolve())
