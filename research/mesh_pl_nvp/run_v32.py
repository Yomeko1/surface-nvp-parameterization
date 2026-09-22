"""v3.2 parameterization with independent, observation-only error audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from . import run_harmonic_local as baseline
from .run_v31_audit import audit, is_oom, train, write_json
from .summarize_v31_audit import read, summarize
from .result_layout import create_result, default_archive, expose_final, resolve_run, source_store, validate_locations


def parse_args(argv=None):
    parser = baseline.build_parser()
    parser.description = "v3.2: H1 affine100 -> Local RQS500 -> wide affine H2 300"
    parser.set_defaults(final_harmonic_iters=300, final_harmonic_max_log_scale=.16,
                        final_harmonic_max_shift=.40, check_interval=50)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--export-format", choices=("auto", "obj", "usda"), default="auto")
    parser.add_argument("--archive-dir", default=None, help="Separate directory for models, logs and raw audits")
    parser.add_argument("--slim-result", default=None, help="Optional same-mesh SLIM output for shared-scale figures")
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


def finish_audit(output, device, *, audit_name="audit", report_name="report", source_store_dir=None):
    output = Path(output)
    audit(output, device, audit_name, detailed_steps={"final_harmonic": (100, 200, 300)},
          source_store_dir=source_store_dir)
    summarize(output, audit_name, report_name, source_store_dir=source_store_dir)
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
    mode = argv.pop(0) if argv and argv[0] in {"run", "train", "audit", "report", "present"} else "run"
    if mode in {"audit", "report", "present"}:
        parser = argparse.ArgumentParser(description=f"v3.2 {mode} saved results")
        parser.add_argument("output", type=Path)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--audit-name", default="audit")
        parser.add_argument("--report-name", default="report")
        parser.add_argument("--output-dir", type=Path, help="Fresh presentation directory (present mode)")
        parser.add_argument("--slim-result", default=None)
        args = parser.parse_args(argv)
        archive = resolve_run(args.output)
        store = source_store(archive)
        if mode == "present":
            if args.output_dir is None:
                parser.error("present requires --output-dir")
            from .result_visualization import present
            if not (archive / "completion.json").is_file():
                raise ValueError("the run has no completed audit")
            create_result(args.output_dir, archive)
            present(archive, args.output_dir, audit_name=args.audit_name,
                    report_name=args.report_name, slim_result=args.slim_result)
            return
        if args.output_dir is not None:
            parser.error("--output-dir is only used with present")
        if mode == "report":
            summarize(archive, args.audit_name, args.report_name, source_store_dir=store)
        else:
            result = finish_audit(archive, args.device, audit_name=args.audit_name,
                                 report_name=args.report_name, source_store_dir=store)
            # Re-audits keep their own completion record and do not overwrite the original presentation.
            if not (archive / "completion.json").exists():
                write_json(archive / "completion.json", result)
            if archive != args.output.resolve() and not (args.output / "summary.json").exists():
                from .result_visualization import present
                present(archive, args.output, audit_name=args.audit_name,
                        report_name=args.report_name, slim_result=args.slim_result)
                write_json(args.output / "completion.json", result)
        return
    args = parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    archive = Path(args.archive_dir).resolve() if args.archive_dir else default_archive(output)
    validate_locations(output, archive)
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    create_result(output, archive)
    args.result_dir = str(output)
    args.output_dir = str(archive)
    args.archive_dir = str(archive)
    args.source_store = str(source_store(archive))
    phase = "training"
    try:
        train(args, argparse.Namespace(snapshot_interval=args.snapshot_interval))
        expose_final(archive, output)
        if mode == "train":
            write_json(output / "completion.json", {"status": "training_complete", "audit": "pending"})
        if mode == "run":
            phase = "audit"
            result = finish_audit(archive, args.device, source_store_dir=args.source_store)
            write_json(archive/"completion.json", result)
            phase = "presentation"
            from .result_visualization import present
            present(archive, output, slim_result=args.slim_result)
            write_json(output/"completion.json", result)
    except Exception as exc:
        if output.exists():
            expose_final(archive, output)
            write_json(output/"completion.json", {"status": "oom" if is_oom(exc) else "failed",
                "phase": phase, "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
