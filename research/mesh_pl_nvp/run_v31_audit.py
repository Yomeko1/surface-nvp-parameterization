"""Reproduce v3.1 training; audit saved native-precision states independently."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import psutil
import torch

from surface_nvp.io import MeshData

from . import run_harmonic_local as baseline
from .audit_metrics import geometric_roundtrip, network_residual, scales, sd_metrics, validity


BASE = "0de8cbbef1f1842bdac54d8dba5662c78a749f06"


def is_oom(exc):
    return isinstance(exc, (torch.OutOfMemoryError, MemoryError)) or any(
        text in str(exc).lower() for text in ("out of memory", "not enough memory", "cannot allocate memory"))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def json_safe(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def archive_sources(path, store=None):
    root = Path(__file__).resolve().parents[2]
    target = io.BytesIO() if store is not None else path
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in (root/"research"/"mesh_pl_nvp", root/"surface_nvp"):
            for file in sorted(folder.rglob("*")):
                if file.suffix in {".py", ".yaml", ".md", ".bib", ".txt"}:
                    archive.write(file, file.relative_to(root))
    if store is not None:
        from .result_layout import store_source_bytes
        return store_source_bytes(path, target.getvalue(), store)
    return Path(path)


def rng_state():
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def prepare(args):
    mesh = baseline.load_mesh(args.input, prim_path=args.prim_path)
    topology = baseline.validate_disk_topology(mesh.faces, len(mesh.vertices))
    uv0 = baseline.generate_initial_uv(mesh, method="tutte", boundary_mode="circle", geometry_scale=True).uv
    v, f, u = (torch.as_tensor(x, dtype=d) for x, d in (
        (mesh.vertices, torch.float64), (mesh.faces, torch.long), (uv0, torch.float64)))
    if args.scaffold_rings == 1:
        scaffold = baseline.build_outer_scaffold(v, f, u, scale=args.scaffold_scale)
    else:
        scaffold = baseline.build_multiring_outer_scaffold(v, f, u, scale=args.scaffold_scale,
            rings=args.scaffold_rings, transition_exponent=args.scaffold_transition_exponent)
    rings = weights = None
    if args.scaffold_mode_profile == "geometric":
        rings, weights = baseline.multiring_transition_profile(scaffold,
            rings=args.scaffold_rings, transition_exponent=args.scaffold_transition_exponent)
    names, modes = baseline.build_radial_fourier_harmonic_modes(scaffold.uv, scaffold.faces,
        scaffold.original_boundary, scaffold.outer_boundary, frequencies=args.frequencies,
        transition_rings=rings, transition_weights=weights)
    counts, support = baseline.select_supported_boundary_hat_counts(scaffold.uv,
        scaffold.original_boundary, args.boundary_hat_counts,
        minimum_peak_weight=args.boundary_hat_min_peak_weight)
    counts = counts if args.auto_boundary_hat_counts else args.boundary_hat_counts
    if counts:
        hn, hm = baseline.build_piecewise_linear_boundary_harmonic_modes(scaffold.uv,
            scaffold.faces, scaffold.original_boundary, scaffold.outer_boundary,
            control_counts=counts, transition_rings=rings, transition_weights=weights)
        names, modes = names+hn, torch.cat((modes, hm), dim=0)
    preflight = baseline.preflight_mode_intervals(scaffold.uv, scaffold.faces, modes, names,
                                                 area_margin_ratio=args.area_margin_ratio)
    model_kwargs = {k: getattr(args, k) for k in (
        "harmonic_cycles", "harmonic_max_log_scale", "harmonic_max_shift", "area_margin_ratio",
        "local_cycles", "local_hidden_dim", "local_radial_map", "local_max_log_scale",
        "local_max_shift_fraction", "local_center_iterations", "local_latent_transform",
        "local_spline_bins", "local_spline_bound")}
    for key in ("final_harmonic_max_log_scale", "final_harmonic_max_shift"):
        if getattr(args, key, None) is not None:
            model_kwargs[key] = getattr(args, key)
    model = baseline.HarmonicLocalHarmonicFlow(scaffold.vertices_3d, scaffold.faces,
        scaffold.uv, modes, names, **model_kwargs).to(device=args.device, dtype=torch.float64)
    bundle = {"vertices": v, "faces": f, "initial_original_uv": u,
              "extended_vertices": scaffold.vertices_3d, "extended_faces": scaffold.faces,
              "reference": scaffold.uv, "modes": modes, "names": names,
              "boundary": scaffold.original_boundary, "model_kwargs": model_kwargs,
              "scales": scales(mesh.vertices, uv0), "config": vars(args),
              "mesh_prim_path": mesh.prim_path}
    return model, bundle, mesh, {"topology": topology, "preflight": preflight,
                               "effective_hat_counts": list(counts), "support": support}


class Recorder:
    def __init__(self, output, model, phase, iterations, interval=10):
        self.output, self.model, self.phase = Path(output), model, phase
        self.iterations, self.interval = iterations, interval
        self.start = time.perf_counter()
        self.overhead = 0.
        self.optimizer = None
        self.last_step = None

    def append(self, filename, row):
        with (self.output/filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False)+"\n")

    def __call__(self, step, mapped, loss, diagnostics, optimizer):
        start = time.perf_counter()
        self.optimizer, self.last_step = optimizer, step
        with torch.no_grad():
            area = baseline.signed_double_areas(mapped, self.model.faces)/2
            row = {"phase": self.phase, "step": step, "sd": float(loss.detach()),
                   "lr": float(optimizer.param_groups[0]["lr"]),
                   "minimum_extended_area": float(area.min()),
                   "q_max": float(diagnostics.q_values.detach().max()),
                   "wall_seconds": time.perf_counter()-self.start,
                   "cuda_peak_allocated": torch.cuda.max_memory_allocated() if mapped.is_cuda else None,
                   "process_rss": psutil.Process().memory_info().rss}
            self.append("training.jsonl", row)
            if step % self.interval == 0 or step == self.iterations:
                payload = {"phase": self.phase, "step": step, "uv": cpu_copy(mapped),
                           "parameters": {k: cpu_copy(v) for k, v in self.model.named_parameters()},
                           "optimizer": cpu_copy(optimizer.state_dict()), "rng": rng_state(),
                           "schedule": cpu_copy(getattr(self, "schedule_state", None)),
                           "training": row}
                torch.save(payload, self.output/"snapshots"/f"{self.phase}_{step:04d}.pt")
                print(json.dumps(row), flush=True)
        self.overhead += time.perf_counter()-start

    def gradient(self, step, norm):
        self.append("gradients.jsonl", {"phase": self.phase, "step": step,
                                        "preclip_l2": float(norm.detach())})


def manifest(args, extra):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    versions = {}
    for name in ("torch", "numpy", "scipy", "nflows", "shapely", "usd-core", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {}
    for folder in (root/"research"/"mesh_pl_nvp", root/"surface_nvp"):
        for path in folder.rglob("*.py"):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"base_commit": BASE, "release": getattr(args, "release_version", "v3.1-audit"),
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"), "git_status": git("status", "--short"),
            "started_at": datetime.now().astimezone().isoformat(), "config": vars(args),
            "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            "source_sha256": sources, "versions": versions, "python": sys.version,
            "platform": platform.platform(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "memory_total": psutil.virtual_memory().total, "audit": extra,
            "error_thresholds": None, "rollback": False,
            "baseline_risk_schedule_and_area_assertions_preserved": True}


def train(args, options):
    baseline.validate_args(args)
    if options.snapshot_interval < 1:
        raise ValueError("snapshot interval must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/"snapshots").mkdir()
    write_json(output/"manifest.json", manifest(args, vars(options)))
    archive_sources(output/"source.zip", getattr(args, "source_store", None))
    start = time.perf_counter()
    model = bundle = recorder = None
    phase = "preprocessing"
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        model, bundle, mesh, info = prepare(args)
        torch.save(bundle, output/"reference.pt")
        write_json(output/"preprocessing.json", info)
        np.savez_compressed(output/"initial.npz", uv=bundle["reference"].numpy())
        reference = model.initial_uv
        current = reference
        v, f, boundary = (bundle[k].to(args.device) for k in ("vertices", "faces", "boundary"))
        histories, times = {}, {"preprocessing": time.perf_counter()-start}
        offset = 0
        for phase, count in (("first_harmonic", args.harmonic_iters),
                             ("local", args.local_iters), ("final_harmonic", args.final_harmonic_iters)):
            model.set_trainable_stage(phase)
            torch.save(cpu_copy(current), output/f"{phase}_input.pt")
            recorder = Recorder(output, model, phase, count, options.snapshot_interval)
            common = dict(iterations=count, check_interval=args.check_interval,
                          gradient_clip=args.gradient_clip, iteration_offset=offset, observer=recorder)
            if phase == "local":
                kwargs = {k.removeprefix("local_"): getattr(args, k) for k in vars(args)
                          if k.startswith("local_risk_")}
                kwargs["risk_adaptive"] = args.local_risk_adaptive
                kwargs.update(dynamic_risk_threshold=args.local_dynamic_risk_threshold,
                              dynamic_floor_lr=args.local_dynamic_floor_lr)
                current, history, seconds, lr_info = baseline._train_local_stage(model.local,
                    current, v, f, reference, boundary, learning_rate=args.local_lr, **common, **kwargs)
                write_json(output/"local_lr.json", lr_info)
            else:
                current, history, seconds = baseline._train_harmonic_stage(getattr(model, phase),
                    current, v, f, reference, boundary, learning_rate=args.harmonic_lr,
                    phase=phase, **common)
            histories[phase] = history
            times[phase] = {"wall_seconds": seconds, "recording_seconds": recorder.overhead}
            write_json(output/"history.json", histories)
            write_json(output/"timings.json", times)
            np.savez_compressed(output/f"{phase}_final.npz", uv=current.detach().cpu().numpy())
            offset += count
        export_format = getattr(args, "export_format", "usda")
        baseline.save_mesh(output/f"final.{export_format}", mesh,
                           uv=current.detach().cpu().numpy()[:len(mesh.vertices)])
        torch.save({"format_version": 1, "reference": bundle,
                    "parameters": {key: cpu_copy(value) for key, value in model.named_parameters()},
                    "uv": cpu_copy(current)}, output/"final.model.pt")
        write_json(output/"status.json", {"status": "training_complete", "updates": offset,
                                          "wall_seconds": time.perf_counter()-start})
    except Exception as exc:
        status = {"status": "oom" if is_oom(exc) else "failed",
                  "phase": phase, "last_observed_step": recorder.last_step if recorder else None,
                  "error": repr(exc), "traceback": traceback.format_exc(),
                  "wall_seconds": time.perf_counter()-start}
        write_json(output/"status.json", status)
        if model is not None:
            torch.save({"parameters": {k: cpu_copy(v) for k, v in model.named_parameters()},
                        "optimizer": cpu_copy(recorder.optimizer.state_dict()) if recorder and recorder.optimizer else None,
                        "rng": rng_state()}, output/"failure_state.pt")
        print(json.dumps(status), flush=True)
        raise


def load_model(bundle, device):
    return baseline.HarmonicLocalHarmonicFlow(bundle["extended_vertices"], bundle["extended_faces"],
        bundle["reference"], bundle["modes"], bundle["names"], **bundle["model_kwargs"]).to(device=device, dtype=torch.float64)


def load_checkpoint(path, device="cpu"):
    """Load only trusted project checkpoints; torch pickle data is executable."""
    if Path(path).is_dir():
        from .result_layout import resolve_run
        path = resolve_run(path)/"final.model.pt"
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if payload["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    model = load_model(payload["reference"], device)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            parameter.copy_(payload["parameters"][key])
    return model, payload


def audit(output, device, audit_name="audit", detailed_steps=None, source_store_dir=None):
    output = Path(output)
    if Path(audit_name).name != audit_name or audit_name in {".", ".."}:
        raise ValueError("audit name must be a simple subdirectory name")
    directory = output/audit_name
    directory.mkdir(exist_ok=False)
    audit_start = time.perf_counter()
    source_archive = archive_sources(directory/"source.zip", source_store_dir)
    write_json(directory/"manifest.json", {"started_at": datetime.now().astimezone().isoformat(),
        "device": device, "torch": torch.__version__, "numpy": np.__version__,
        "shapely": importlib.metadata.version("shapely"),
        "reference_sha256": hashlib.sha256((output/"reference.pt").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest()})
    bundle = torch.load(output/"reference.pt", weights_only=False, map_location="cpu")
    model = load_model(bundle, device)
    reference = bundle["reference"].numpy()
    vertices, faces = bundle["vertices"].numpy(), bundle["faces"].numpy()
    extended_faces = bundle["extended_faces"].numpy()
    normalizers = bundle["scales"]
    export_format = bundle["config"].get("export_format", "usda")
    mesh = MeshData(vertices, faces, prim_path=bundle.get("mesh_prim_path"))
    write_json(directory/"scales.json", normalizers)
    files = sorted((output/"snapshots").glob("*.pt"))
    phase_order = {"first_harmonic": 0, "local": 1, "final_harmonic": 2}
    files.sort(key=lambda p: (phase_order[p.stem.rsplit("_", 1)[0]], p.stem.rsplit("_", 1)[1]))
    records = []
    def geometry_record(name, uv, detailed):
        started = time.perf_counter()
        row, arrays = sd_metrics(vertices, faces, uv[:len(vertices)])
        sd_seconds = time.perf_counter()-started
        geometry, sample = geometric_roundtrip(vertices, faces, uv[:len(vertices)], normalizers["D3D"])
        geometry_seconds = time.perf_counter()-started-sd_seconds
        val_orig, pairs_orig = validity(uv[:len(vertices)], faces, intersections=detailed)
        val_ext, pairs_ext = validity(uv, extended_faces, intersections=detailed)
        np.savez_compressed(directory/f"{name}_details.npz", **arrays,
                            **{f"geometry_{k}": v for k, v in sample.items()},
                            original_intersections=pairs_orig, extended_intersections=pairs_ext)
        result = {"sd": row, "geometry": geometry, "original_validity": val_orig,
                  "extended_validity": val_ext,
                  "timings": {"sd_seconds": sd_seconds, "geometry_seconds": geometry_seconds,
                              "validity_and_save_seconds": time.perf_counter()-started-sd_seconds-geometry_seconds}}
        if detailed:
            edge, edge_data = geometric_roundtrip(vertices, faces, uv[:len(vertices)], normalizers["D3D"], near_edges=True)
            quant, quant_data = geometric_roundtrip(vertices, faces, uv[:len(vertices)], normalizers["D3D"], quantize_queries=True)
            result.update(geometry_near_edges=edge, geometry_unit_square_float32_query=quant)
            np.savez_compressed(directory/f"{name}_extra_geometry.npz",
                **{f"edge_{k}": v for k, v in edge_data.items()}, **{f"quant_{k}": v for k, v in quant_data.items()})
            export_path = directory/f"{name}.{export_format}"
            baseline.save_mesh(export_path, mesh, uv=uv[:len(vertices)])
            exported = baseline.load_mesh(export_path, prim_path=mesh.prim_path)
            result["export_format"] = export_format
            if not np.array_equal(exported.faces, faces):
                raise ValueError("export changed topology")
            export_geo, export_data = geometric_roundtrip(vertices, faces, uv[:len(vertices)], normalizers["D3D"],
                decode_uv=exported.uv, decode_vertices=exported.vertices)
            result["geometry_export_decode"] = export_geo
            result["export_validity"], export_pairs = validity(exported.uv, faces)
            np.savez_compressed(directory/f"{name}_export.npz", **export_data, intersections=export_pairs,
                                uv=exported.uv, vertices=exported.vertices)
        write_json(directory/f"{name}_geometry.json", result)
        result["timings"]["total_seconds"] = time.perf_counter()-started
        return result
    initial = geometry_record("initial", reference, True)
    write_json(directory/"initial.json", initial)
    for path in files:
        snapshot = torch.load(path, weights_only=False, map_location="cpu")
        phase, step, uv = snapshot["phase"], snapshot["step"], snapshot["uv"].numpy()
        detailed = step == bundle["config"][{"first_harmonic": "harmonic_iters", "local": "local_iters",
                                              "final_harmonic": "final_harmonic_iters"}[phase]]
        detailed = detailed or step in (detailed_steps or {}).get(phase, ())
        record = {"phase": phase, "step": step, "snapshot": path.name,
                  **geometry_record(path.stem, uv, detailed)}
        with torch.no_grad():
            for k, parameter in model.named_parameters():
                parameter.copy_(snapshot["parameters"][k])
            sequence = [model.first_harmonic]
            if phase != "first_harmonic":
                sequence.append(model.local)
            if phase == "final_harmonic":
                sequence.append(model.final_harmonic)
            for kind in ("composed", "isolated"):
                try:
                    target = reference if kind == "composed" else torch.load(
                        output/f"{phase}_input.pt", weights_only=True).numpy()
                    active = sequence if kind == "composed" else sequence[-1:]
                    reproduced = torch.as_tensor(target, device=device)
                    for module in active:
                        reproduced = module(reproduced)
                    cached_delta = reproduced.cpu().numpy()-uv
                    record[f"{kind}_recomputed_vs_cached_max_abs"] = float(np.abs(cached_delta).max())
                    restored = reproduced
                    for module in reversed(active):
                        restored = module(restored, inverse=True)
                    metric, delta = network_residual(restored.cpu().numpy(), target, len(vertices), normalizers["LUV0"])
                    record[f"network_{kind}"] = {
                        "status": "nonfinite" if not np.isfinite(delta).all() else "ok", **metric}
                    np.save(directory/f"{path.stem}_network_{kind}.npy", delta)
                except Exception as exc:
                    if is_oom(exc):
                        raise
                    record[f"network_{kind}"] = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
        records.append(record)
        write_json(directory/f"{path.stem}.json", record)
        write_json(directory/"summary.json", records)
        print(f"audited {path.stem}: SD={record['sd']['regularized_f64']['area_weighted_mean']:.9g}", flush=True)
    write_json(directory/"status.json", {"status": "complete", "snapshots": len(records),
                                         "wall_seconds": time.perf_counter()-audit_start})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "audit"))
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--audit-output")
    parser.add_argument("--audit-device", default="cuda")
    parser.add_argument("--audit-name", default="audit")
    options, rest = parser.parse_known_args()
    if options.mode == "train":
        if options.snapshot_interval < 1:
            raise ValueError("snapshot interval must be positive")
        train(baseline.parse_args(rest), options)
    else:
        try:
            audit(options.audit_output, options.audit_device, options.audit_name)
        except Exception as exc:
            output = Path(options.audit_output)
            failure = output/("audit_failure_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".json")
            write_json(failure, {"status": "oom" if is_oom(exc) else "failed",
                                 "error": repr(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
