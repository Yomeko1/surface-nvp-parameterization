from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.training.summary import SUMMARY_FIELDS, build_run_summary
from surface_nvp.training.config import load_config


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Balls": ROOT / "data/input/Balls/Balls.obj",
    "David328": ROOT / "data/input/David328/David328.usda",
    "NefertitiFace": ROOT / "data/input/NefertitiFace/NefertitiFace.usda",
    "Cow": ROOT / "data/input/Cow/Cow_dABF.usda",
    "Isis": ROOT / "data/input/Isis/Isis_dABF.usda",
    "00027": ROOT / "data/input/00027/Input.obj",
}
METHODS = ("direct", "affine", "spline", "slim")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible parameterization comparison matrix")
    parser.add_argument("--output-root", default="data/output/benchmark")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--init-method", choices=["tutte", "mean_value", "abfpp", "auto"], default=None)
    parser.add_argument("--init-boundary", choices=["circle", "square"], default=None)
    parser.add_argument("--abfpp-executable", default=None)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--slim-iters", type=int, default=20)
    parser.add_argument("--check-interval", type=int, default=25)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument("--spline-bins", type=int, default=None)
    parser.add_argument("--mixing-type", choices=["none", "rotation"], default=None)
    parser.add_argument("--global-transform", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default=None)
    parser.add_argument("--lbfgs-iters", type=int, default=None)
    parser.add_argument("--lbfgs-lr", type=float, default=None)
    parser.add_argument("--lbfgs-check-interval", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-device", default="cuda")
    parser.add_argument("--intersection-batch-size", type=int, default=65536)
    parser.add_argument("--slim-executable", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--stop-after-dataset", choices=DATASETS, default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [0 if args.seed is None else args.seed]
    multi_seed = args.seeds is not None
    if any(seed < 0 for seed in seeds):
        parser.error("seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    deterministic_methods = set(args.methods) - {"affine", "spline"}
    if multi_seed and deterministic_methods:
        parser.error(
            "--seeds supports stochastic affine/spline methods only; run deterministic "
            f"methods once with --seed ({', '.join(sorted(deterministic_methods))})"
        )
    if min(args.iters, args.slim_iters, args.check_interval, args.intersection_batch_size) <= 0:
        parser.error("iteration counts, check interval, and batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    slim_executable = Path(args.slim_executable).resolve() if args.slim_executable else None
    if "slim" in args.methods and (slim_executable is None or not slim_executable.is_file()):
        parser.error("--slim-executable must point to the built SLIM executable")

    config_path = (ROOT / args.config).resolve()
    config = load_config(str(config_path))
    geometry_scale = config["init"]["geometry_scale"]
    args.init_method = args.init_method or config["init"]["method"]
    args.init_boundary = args.init_boundary or config["init"]["boundary"]
    configured_abfpp = args.abfpp_executable or config["init"]["abfpp_executable"]
    abfpp_executable = (ROOT / configured_abfpp).resolve() if configured_abfpp else None
    if args.init_method == "abfpp" and (abfpp_executable is None or not abfpp_executable.is_file()):
        parser.error("ABF++ initialization requires --abfpp-executable")
    for key in ("num_layers", "hidden_dim", "mlp_layers", "spline_bins", "mixing_type", "global_transform"):
        if getattr(args, key) is None:
            setattr(args, key, config["model"][key])
    for key in ("lr", "lr_schedule", "lbfgs_iters", "lbfgs_lr", "lbfgs_check_interval"):
        if getattr(args, key) is None:
            setattr(args, key, config["train"][key])
    if args.lbfgs_iters < 0 or min(args.lr, args.lbfgs_lr, args.lbfgs_check_interval) <= 0:
        parser.error("learning rates and L-BFGS check interval must be positive; L-BFGS iterations must be non-negative")
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = _manifest(
        args,
        seeds,
        multi_seed,
        output_root,
        slim_executable,
        abfpp_executable,
        config_path,
        geometry_scale,
    )
    existing_manifest = _read_json(manifest_path)
    if existing_manifest is not None and existing_manifest.get("run_signature") != manifest["run_signature"]:
        parser.error("output root belongs to a different benchmark configuration; choose a new directory")
    if existing_manifest is None:
        _write_json(manifest_path, manifest)
    else:
        manifest = existing_manifest

    state_path = output_root / "runs.json"
    state = _read_json(state_path, default={"runs": []})
    records = {record["key"]: record for record in state.get("runs", [])}
    initial_hashes = manifest.setdefault("initial_uv_sha256", {})
    initial_info = manifest.setdefault("initial_uv_info", {})

    for dataset in args.datasets:
        input_path = DATASETS[dataset]
        initial_path = output_root / dataset / "initial" / f"{dataset}_initial{input_path.suffix}"
        expected_initial_hash = initial_hashes.get(dataset)
        if initial_path.is_file() and not args.force:
            actual_initial_hash = _sha256_file(initial_path)
            if expected_initial_hash != actual_initial_hash:
                raise RuntimeError(
                    f"shared initial UV does not match the manifest: {initial_path}; use a new output root"
                )
        else:
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            init_command = [
                    sys.executable,
                    str(ROOT / "scripts/init_uv.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(initial_path),
                    "--method",
                    args.init_method,
                    "--boundary",
                    args.init_boundary,
                    _geometry_scale_flag(geometry_scale),
                ]
            if abfpp_executable is not None:
                init_command.extend(["--abfpp-executable", str(abfpp_executable)])
            _run_checked(init_command, initial_path.with_suffix(".log"))
            actual_initial_hash = _sha256_file(initial_path)
            if expected_initial_hash is not None and expected_initial_hash != actual_initial_hash and not args.force:
                raise RuntimeError(f"regenerated initial UV differs from the manifest: {initial_path}")
            initial_hashes[dataset] = actual_initial_hash
            info_path = initial_path.with_suffix(".init.json")
            initial_info[dataset] = _read_json(info_path) if info_path.is_file() else None
            _write_json(manifest_path, manifest)

        for seed in seeds:
            for method in args.methods:
                key = _run_key(dataset, method, seed, manifest["run_signature"])
                output_path = _output_path(
                    output_root, dataset, method, seed, input_path.suffix, multi_seed
                )
                metrics_path = output_path.with_suffix(".metrics.json")
                record = records.get(key)
                expected_metrics = _display_path(metrics_path)
                if not args.force and _is_completed_record(
                    record, metrics_path, expected_metrics, initial_hashes[dataset]
                ):
                    print(f"skip completed {key}", flush=True)
                    continue
                if metrics_path.is_file():
                    metrics_path.unlink()

                command = _method_command(
                    args,
                    method,
                    seed,
                    input_path,
                    initial_path,
                    output_path,
                    slim_executable,
                    geometry_scale,
                )
                print(f"run {key}", flush=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
                wall_elapsed = time.perf_counter() - start
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.with_suffix(".log").write_text(
                    f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
                    encoding="utf-8",
                )
                records[key] = {
                    "key": key,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "run_signature": manifest["run_signature"],
                    "initial_uv_sha256": initial_hashes[dataset],
                    "returncode": completed.returncode,
                    "wall_elapsed_seconds": wall_elapsed,
                    "metrics": _display_path(metrics_path) if metrics_path.is_file() else None,
                    "metrics_sha256": _sha256_file(metrics_path) if metrics_path.is_file() else None,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_json(state_path, {"runs": list(records.values())})
                print(f"finished {key} in {wall_elapsed:.1f}s (exit {completed.returncode})", flush=True)
                if completed.returncode != 0 and not args.continue_on_error:
                    raise RuntimeError(f"benchmark run failed: {key}; see {output_path.with_suffix('.log')}")

        if args.stop_after_dataset == dataset:
            break

    _write_summary(output_root, args, seeds, multi_seed, records)


def _method_command(
    args, method, seed, input_path, initial_path, output_path, slim_executable, geometry_scale
):
    common = [
        "--input",
        str(input_path),
        "--initial-uv",
        str(initial_path),
        "--output",
        str(output_path),
        _geometry_scale_flag(geometry_scale),
    ]
    if method == "slim":
        return [
            sys.executable,
            str(ROOT / "scripts/run_slim.py"),
            *common,
            "--executable",
            str(slim_executable),
            "--iters",
            str(args.slim_iters),
        ]

    script = "train_direct_uv.py" if method == "direct" else "train_injective_nvp.py"
    command = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "--config",
        str((ROOT / args.config).resolve()),
        *common,
        "--seed",
        str(seed),
        "--iters",
        str(args.iters),
        "--check-interval",
        str(args.check_interval),
        "--lr",
        str(args.lr),
        "--lr-schedule",
        args.lr_schedule,
        "--lbfgs-iters",
        str(args.lbfgs_iters),
        "--lbfgs-lr",
        str(args.lbfgs_lr),
        "--lbfgs-check-interval",
        str(args.lbfgs_check_interval),
        "--device",
        args.device,
        "--validation-device",
        args.validation_device,
        "--intersection-batch-size",
        str(args.intersection_batch_size),
    ]
    if method in {"affine", "spline"}:
        command.extend(["--coupling-type", method])
        model_overrides = []
        if args.num_layers is not None:
            model_overrides.extend(["--num-layers", str(args.num_layers)])
        if args.hidden_dim is not None:
            model_overrides.extend(["--hidden-dim", str(args.hidden_dim)])
        if args.mlp_layers is not None:
            model_overrides.extend(["--mlp-layers", str(args.mlp_layers)])
        if args.spline_bins is not None:
            model_overrides.extend(["--spline-bins", str(args.spline_bins)])
        model_overrides.extend(["--mixing-type", args.mixing_type])
        model_overrides.append(_global_transform_flag(args.global_transform))
        command.extend(model_overrides)
    return command


def _write_summary(output_root: Path, args, seeds: list[int], multi_seed: bool, records: dict) -> None:
    manifest = _read_json(output_root / "manifest.json")
    run_signature = manifest["run_signature"]
    rows = []
    for dataset in args.datasets:
        expected_initial_hash = manifest["initial_uv_sha256"].get(dataset)
        if expected_initial_hash is None:
            continue
        initial_added = False
        input_suffix = DATASETS[dataset].suffix
        for seed in seeds:
            for method in args.methods:
                key = _run_key(dataset, method, seed, run_signature)
                record = records.get(key, {})
                output_path = _output_path(
                    output_root, dataset, method, seed, input_suffix, multi_seed
                )
                metrics_path = output_path.with_suffix(".metrics.json")
                if not _metrics_match_record(
                    record,
                    metrics_path,
                    _display_path(metrics_path),
                    expected_initial_hash,
                ):
                    iters = args.slim_iters if method == "slim" else args.iters
                    rows.append(
                        _missing_summary_row(dataset, method, seed, iters, record, metrics_path)
                    )
                    continue
                payload = _read_json(metrics_path)
                if not initial_added:
                    initial_payload = {
                        "initial": payload["initial"],
                        "final": payload["initial"],
                        "training": {"selected_iteration": 0},
                        "history": [],
                    }
                    initial_seed = None if multi_seed else seed
                    rows.append(
                        _summary_row(
                            dataset, "initial", initial_seed, 0, initial_payload, {}, metrics_path
                        )
                    )
                    initial_added = True
                iters = args.slim_iters if method == "slim" else args.iters
                rows.append(_summary_row(dataset, method, seed, iters, payload, record, metrics_path))

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
    _write_json(output_root / "summary.json", rows)


def _summary_row(dataset, method, seed, iters, payload, record, metrics_path):
    row = build_run_summary(method, iters, payload)
    returncode = record.get("returncode")
    complete = returncode in (None, 0) and row["valid"]
    row.update(
        {
            "dataset": dataset,
            "seed": seed,
            "status": "complete" if complete else "failed",
            "wall_elapsed_seconds": record.get("wall_elapsed_seconds"),
            "training_elapsed_seconds": (payload.get("training") or {}).get("elapsed_seconds"),
            "source": _display_path(metrics_path),
        }
    )
    return row


def _missing_summary_row(dataset, method, seed, iters, record, metrics_path):
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(
        {
            "dataset": dataset,
            "seed": seed,
            "status": "failed" if record else "missing",
            "wall_elapsed_seconds": record.get("wall_elapsed_seconds"),
            "training_elapsed_seconds": None,
            "source": _display_path(metrics_path),
            "method": method,
            "iters": iters,
            "valid": False,
        }
    )
    return row


def _manifest(
    args,
    seeds,
    multi_seed,
    output_root,
    slim_executable,
    abfpp_executable,
    config_path,
    geometry_scale,
):
    packages = {}
    for name in ("numpy", "scipy", "torch", "matplotlib", "PyYAML", "nflows", "usd-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    output_relative = _relative_path(output_root)
    if output_relative is not None:
        output_prefix = output_relative.as_posix().rstrip("/") + "/"
        git_status = [line for line in git_status if not line[3:].replace("\\", "/").startswith(output_prefix)]
    dataset_hashes = {name: _sha256_file(DATASETS[name]) for name in args.datasets}
    slim_hash = _sha256_file(slim_executable) if slim_executable else None
    abfpp_hash = _sha256_file(abfpp_executable) if abfpp_executable else None
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_driver": _gpu_driver_version(),
    }
    seed_settings = {"seeds": seeds} if multi_seed else {"seed": seeds[0]}
    signature_payload = {
        "git_commit": git_commit,
        "git_status": git_status,
        "source_sha256": _source_sha256(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "datasets": dataset_hashes,
        "methods": list(args.methods),
        "init_method": args.init_method,
        "init_boundary": args.init_boundary,
        **seed_settings,
        "iters": args.iters,
        "slim_iters": args.slim_iters,
        "check_interval": args.check_interval,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "mlp_layers": args.mlp_layers,
        "spline_bins": args.spline_bins,
        "mixing_type": args.mixing_type,
        "global_transform": args.global_transform,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lbfgs_iters": args.lbfgs_iters,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_check_interval": args.lbfgs_check_interval,
        "device": args.device,
        "validation_device": args.validation_device,
        "intersection_batch_size": args.intersection_batch_size,
        "geometry_scale": geometry_scale,
        "slim_sha256": slim_hash,
        "abfpp_sha256": abfpp_hash,
        "initial_uv_sha256": {},
        "environment": environment,
    }
    run_signature = _benchmark_signature(signature_payload)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_signature": run_signature,
        "git_commit": git_commit,
        "git_status": git_status,
        **environment,
        "source_sha256": signature_payload["source_sha256"],
        "output_root": _display_path(output_root),
        "datasets": {
            name: {"path": _display_path(DATASETS[name]), "sha256": dataset_hashes[name]}
            for name in args.datasets
        },
        "methods": args.methods,
        "init_method": args.init_method,
        "init_boundary": args.init_boundary,
        **seed_settings,
        "iters": args.iters,
        "slim_iters": args.slim_iters,
        "check_interval": args.check_interval,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "mlp_layers": args.mlp_layers,
        "spline_bins": args.spline_bins,
        "mixing_type": args.mixing_type,
        "global_transform": args.global_transform,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lbfgs_iters": args.lbfgs_iters,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_check_interval": args.lbfgs_check_interval,
        "device": args.device,
        "validation_device": args.validation_device,
        "intersection_batch_size": args.intersection_batch_size,
        "config": _display_path(config_path),
        "config_sha256": signature_payload["config_sha256"],
        "geometry_scale": geometry_scale,
        "slim_executable": str(slim_executable) if slim_executable else None,
        "slim_sha256": slim_hash,
        "abfpp_executable": str(abfpp_executable) if abfpp_executable else None,
        "abfpp_sha256": abfpp_hash,
    }
    return manifest


def _geometry_scale_flag(enabled: bool) -> str:
    return "--geometry-scale" if enabled else "--no-geometry-scale"


def _global_transform_flag(enabled: bool) -> str:
    return "--global-transform" if enabled else "--no-global-transform"


def _gpu_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def _run_key(dataset: str, method: str, seed: int, run_signature: str) -> str:
    return f"{dataset}:{method}:seed{seed}:{run_signature[:12]}"


def _output_path(
    output_root: Path, dataset: str, method: str, seed: int, suffix: str, multi_seed: bool
) -> Path:
    if multi_seed:
        return output_root / dataset / f"seed_{seed}" / method / f"{dataset}_{method}_seed{seed}{suffix}"
    return output_root / dataset / method / f"{dataset}_{method}{suffix}"


def _benchmark_signature(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_completed_record(
    record: dict | None, metrics_path: Path, expected_metrics: str, expected_initial_hash: str
) -> bool:
    return bool(
        record
        and record.get("returncode") == 0
        and _metrics_match_record(record, metrics_path, expected_metrics, expected_initial_hash)
    )


def _metrics_match_record(
    record: dict | None, metrics_path: Path, expected_metrics: str, expected_initial_hash: str
) -> bool:
    return bool(
        record
        and metrics_path.is_file()
        and record.get("metrics") == expected_metrics
        and record.get("metrics_sha256") == _sha256_file(metrics_path)
        and record.get("initial_uv_sha256") == expected_initial_hash
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    roots = [
        ROOT / "surface_nvp",
        ROOT / "scripts",
        ROOT / "external/slim_runner",
        ROOT / "external/abfpp_runner",
        ROOT / "configs",
    ]
    files = [ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.as_posix()):
        digest.update(_display_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _relative_path(path: Path) -> Path | None:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return None


def _display_path(path: Path) -> str:
    relative = _relative_path(path)
    return str(relative if relative is not None else path)


def _run_checked(command, log_path):
    print(f"run {subprocess.list2cmdline(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}")


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


if __name__ == "__main__":
    main()
