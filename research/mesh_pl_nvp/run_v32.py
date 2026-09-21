"""v3.2 parameterization with independent, observation-only error audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from . import run_harmonic_local as baseline
from .run_v31_audit import audit, is_oom, train, write_json
from .summarize_v31_audit import read, summarize


def parse_args(argv=None):
    parser = baseline.build_parser()
    parser.description = "v3.2: H1 affine100 -> Local RQS500 -> wide affine H2 300"
    parser.set_defaults(final_harmonic_iters=300, final_harmonic_max_log_scale=.16,
                        final_harmonic_max_shift=.40, check_interval=50)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--export-format", choices=("auto", "obj", "usda"), default="auto")
    args = baseline.parse_args(argv, parser=parser)
    baseline.validate_args(args)
    if args.snapshot_interval < 1:
        raise ValueError("snapshot-interval must be positive")
    args.input = str(Path(args.input).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.export_format == "auto":
        args.export_format = "obj" if Path(args.input).suffix.lower() == ".obj" else "usda"
    args.release_version = "v3.2"
    return args


def finish_audit(output, device, *, audit_name="audit", report_name="report"):
    output = Path(output)
    audit(output, device, audit_name, detailed_steps={"final_harmonic": (100, 200, 300)})
    summarize(output, audit_name, report_name)
    rows = read(output/audit_name/"summary.json")
    initial = read(output/audit_name/"initial.json")
    config = read(output/"manifest.json")["config"]
    required = {(phase, config[key]) for phase, key in (("first_harmonic", "harmonic_iters"),
                ("local", "local_iters"), ("final_harmonic", "final_harmonic_iters"))}
    complete = (read(output/"status.json")["status"] == "training_complete" and
                required.issubset({(row["phase"], row["step"]) for row in rows}))
    checked = [value for row in rows for key in ("original_validity", "extended_validity", "export_validity")
               if (value := row.get(key)) is not None and value["intersection_status"] != "not_requested"]
    checked.extend(initial[key] for key in ("original_validity", "extended_validity", "export_validity"))
    valid = complete and all(value["intersection_status"] == "complete" and
                not any(value[key] for key in ("flipped", "degenerate", "nonfinite_faces", "intersection_pairs"))
                for value in checked)
    result = {"status": "complete" if valid else "invalid_geometry", "release": "v3.2",
        "audit_name": audit_name, "report_name": report_name,
        "geometry_audits_valid": valid, "training_and_endpoints_complete": complete,
        "audited_snapshots": len(rows),
        "network_non_ok_snapshots": [row["snapshot"] for row in rows if row["network_composed"]["status"] != "ok"],
        "network_residual_threshold": None, "network_ok_means": "finite computation, not an accuracy certificate",
        "rollback": False}
    write_json(output/audit_name/"completion.json", result)
    if not valid:
        raise RuntimeError("geometric validity audit failed; diagnostic artifacts retained")
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv.pop(0) if argv and argv[0] in {"run", "train", "audit", "report"} else "run"
    if mode in {"audit", "report"}:
        parser = argparse.ArgumentParser(description=f"v3.2 {mode} saved results")
        parser.add_argument("output", type=Path)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--audit-name", default="audit")
        parser.add_argument("--report-name", default="report")
        args = parser.parse_args(argv)
        if mode == "report":
            summarize(args.output, args.audit_name, args.report_name)
        else:
            finish_audit(args.output, args.device, audit_name=args.audit_name, report_name=args.report_name)
        return
    args = parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    phase = "training"
    try:
        train(args, argparse.Namespace(snapshot_interval=args.snapshot_interval))
        if mode == "run":
            phase = "audit"
            result = finish_audit(output, args.device)
            write_json(output/"completion.json", result)
    except Exception as exc:
        if output.exists():
            write_json(output/"completion.json", {"status": "oom" if is_oom(exc) else "failed",
                "phase": phase, "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
