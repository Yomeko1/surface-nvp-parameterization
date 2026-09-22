"""Render saved native-f64 parameterizations without retraining or rescaling UV."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
import numpy as np
import torch

from surface_nvp.io import load_mesh
from .audit_metrics import orientation_signs, sd_metrics, validity
from .result_layout import copy_verified, expose_final, relative_path
from .run_v31_audit import write_json
from .summarize_v31_audit import read, validity_checkpoints


def dataset_name(config):
    path = Path(config["input"])
    return path.parent.name.rstrip("#") if path.stem == "Input" else path.stem


def draw_mesh(ax, uv, faces, *, values=None, cmap=None, norm=None, title=""):
    collection = PolyCollection(uv[faces], edgecolors="#555b60", linewidths=.10,
        facecolors="#f1f4f5" if values is None else None, cmap=cmap, norm=norm)
    if values is not None:
        collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.margins(.035)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.tick_params(labelsize=8)
    return collection


def panels(path, faces, entries, *, cmap=None, norm=None, colorbar=None, ticks=None):
    fig, axes = plt.subplots(1, len(entries), figsize=(6.4*len(entries), 5.6),
                             squeeze=False, constrained_layout=True)
    try:
        for ax, (uv, title, values) in zip(axes[0], entries):
            collection = draw_mesh(ax, uv, faces, values=values, cmap=cmap, norm=norm, title=title)
        if colorbar:
            bar = fig.colorbar(collection, ax=axes[0].tolist(), fraction=.035, pad=.025,
                               extend="max" if "clipped" in colorbar else "neither", ticks=ticks)
            bar.set_label(colorbar, fontsize=9)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def sd_title(name, summary):
    stats = summary["regularized_f64"]
    return (f"{name}\nArea-weighted SD={stats['area_weighted_mean']:.6g}; "
            f"face p99={stats['p99_face_count']:.6g}\nMax={stats['max']:.6g}")


def clipped_sd_panels(path, faces, entries):
    values = np.concatenate([entry[2] for entry in entries])
    if not np.isfinite(values).all():
        raise ValueError("cannot render SD with nonfinite face values")
    upper = max(float(np.quantile(values, .95)), 1e-12)
    panels(path, faces, entries, cmap="magma", norm=Normalize(vmin=0, vmax=upper, clip=True),
           colorbar="Regularized face SD (float64); pooled p95-clipped scale")
    return upper


def intersection_counts(faces, pairs):
    counts = np.zeros(len(faces), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs):
        np.add.at(counts, pairs.ravel(), 1)
    return counts


def read_slim(path, vertices, faces, archive):
    mesh = load_mesh(path)
    if not np.array_equal(mesh.faces, faces) or not np.array_equal(mesh.vertices, vertices):
        raise ValueError("SLIM comparison must use exactly the same original geometry and faces")
    if mesh.uv is None:
        raise ValueError("SLIM comparison has no UV")
    metrics, arrays = sd_metrics(vertices, faces, mesh.uv)
    checked, pairs = validity(mesh.uv, faces)
    if checked["intersection_status"] != "complete":
        raise ValueError("SLIM comparison geometry could not be completely audited")
    record = {"input": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "sd": metrics, "validity": checked}
    destination = archive / "presentation_comparison" / record["sha256"]
    saved_arrays = dict(arrays, intersections=pairs)
    if destination.exists():
        previous = read(destination / "slim.json")
        if any(previous[key] != record[key] for key in ("sha256", "sd", "validity")):
            raise ValueError("saved SLIM audit differs; refusing to overwrite")
        with np.load(destination / "slim.npz") as previous_arrays:
            if set(previous_arrays.files) != set(saved_arrays) or any(
                    not np.array_equal(previous_arrays[key], value, equal_nan=True)
                    for key, value in saved_arrays.items()):
                raise ValueError("saved SLIM face data differs; refusing to overwrite")
    else:
        destination.mkdir(parents=True)
        write_json(destination / "slim.json", record)
        np.savez_compressed(destination / "slim.npz", **saved_arrays)
    return mesh.uv, record, arrays, pairs


def present(archive, output, *, audit_name="audit", report_name="report", slim_result=None):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    config = read(archive / "manifest.json")["config"]
    name = dataset_name(config)
    if (output / "summary.json").exists():
        raise FileExistsError("result presentation already exists; choose a fresh result directory")
    output.mkdir(parents=True, exist_ok=True)
    audit = archive / audit_name
    records, initial_record = read(audit / "summary.json"), read(audit / "initial.json")
    key = f"final_harmonic_{config['final_harmonic_iters']:04d}"
    final_record = read(audit / (key + ".json"))
    reference = torch.load(archive / "reference.pt", map_location="cpu", weights_only=False)
    vertices, faces = reference["vertices"].numpy(), reference["faces"].numpy()
    initial_uv = reference["initial_original_uv"].numpy()
    with np.load(archive / "final_harmonic_final.npz") as data:
        final_uv = data["uv"][:len(vertices)].copy()
    with np.load(audit / "initial_details.npz") as data:
        initial_sd = data["sd_regularized"].copy()
    with np.load(audit / (key + "_details.npz")) as data:
        final_sd = data["sd_regularized"].copy()
        pairs = data["original_intersections"].copy()
    if not np.isfinite(final_uv).all() or final_record["original_validity"]["intersection_status"] != "complete":
        raise ValueError("final native geometry has no complete validity audit")
    expose_final(archive, output)
    for extension in ("obj", "usda"):
        initial_mesh = audit / ("initial." + extension)
        if initial_mesh.is_file():
            copy_verified(initial_mesh, output / initial_mesh.name)
    title0, title1 = sd_title(name + " / Initial", initial_record["sd"]), sd_title(name + " / Ours", final_record["sd"])
    panels(output / "initial.uv.png", faces, [(initial_uv, title0, None)])
    panels(output / "final.uv.png", faces, [(final_uv, title1, None)])
    panels(output / "uv_compare.png", faces, [(initial_uv, title0, None), (final_uv, title1, None)])
    limits = {}
    limits["initial.distortion.png"] = clipped_sd_panels(output / "initial.distortion.png", faces,
        [(initial_uv, title0, initial_sd)])
    limits["final.distortion.png"] = clipped_sd_panels(output / "final.distortion.png", faces,
        [(final_uv, title1, final_sd)])
    limits["distortion_compare.png"] = clipped_sd_panels(output / "distortion_compare.png", faces,
        [(initial_uv, title0, initial_sd), (final_uv, title1, final_sd)])
    signs0, _, areas0 = orientation_signs(initial_uv, faces)
    signs1, _, areas1 = orientation_signs(final_uv, faces)
    area_upper = max(float(np.quantile(np.abs(np.concatenate((areas0, areas1))), .95)), 1e-30)
    panels(output / "area_compare.png", faces,
        [(initial_uv, name + " / Initial signed area", np.clip(areas0, -area_upper, area_upper)),
         (final_uv, name + " / Final signed area", np.clip(areas1, -area_upper, area_upper))],
        cmap="RdYlGn", norm=TwoSlopeNorm(vcenter=0, vmin=-area_upper, vmax=area_upper),
        colorbar="Signed UV face area; shared pooled p95-clipped scale")
    for prefix, uv, signs, record in (("initial", initial_uv, signs0, initial_record),
                                       ("final", final_uv, signs1, final_record)):
        val = record["original_validity"]
        panels(output / (prefix + ".flip_heatmap.png"), faces,
            [(uv, f"{name} / {prefix}: flipped={val['flipped']}, degenerate={val['degenerate']}", signs)],
            cmap="RdYlGn", norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 256), ticks=[-1, 0, 1],
            colorbar="Robust orientation sign: -1 flipped, 0 degenerate, +1 positive")
    counts = intersection_counts(faces, pairs)
    panels(output / "final.intersection_heatmap.png", faces,
        [(final_uv, f"{name} / Ours: {len(pairs)} illegal intersection pairs", counts)],
        cmap="YlOrRd", norm=Normalize(vmin=0, vmax=max(int(counts.max()), 1)),
        colorbar="Illegal intersection pairs involving each face")
    offsets = {"first_harmonic": 0, "local": config["harmonic_iters"],
               "final_harmonic": config["harmonic_iters"]+config["local_iters"]}
    training = [json.loads(line) for line in (archive / "training.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    try:
        ax.plot([offsets[r["phase"]]+r["step"] for r in training], [r["sd"] for r in training], color="#177a57")
        ax.set(title=name + " / Area-weighted regularized SD", xlabel="Completed parameter updates", ylabel="SD")
        ax.grid(alpha=.2)
        for boundary in (config["harmonic_iters"], config["harmonic_iters"]+config["local_iters"]):
            ax.axvline(boundary, color="#777777", linewidth=.7)
        fig.savefig(output / "sd_curve.png", dpi=160)
    finally:
        plt.close(fig)
    copy_verified(archive / report_name / "four_metrics.png", output / "four_metrics.png")
    slim = None
    if slim_result is None:
        candidate = output.parent / "slim" / (name + ".obj")
        slim_result = candidate if candidate.is_file() else None
    if slim_result is not None:
        suv, slim, sarrays, spairs = read_slim(slim_result, vertices, faces, archive)
        stitle = sd_title(name + " / SLIM", slim["sd"])
        panels(output / "slim_uv_compare.png", faces, [(suv, stitle, None), (final_uv, title1, None)])
        limits["slim_distortion_compare.png"] = clipped_sd_panels(output / "slim_distortion_compare.png", faces,
            [(suv, stitle, sarrays["sd_regularized"]), (final_uv, title1, final_sd)])
        scounts = intersection_counts(faces, spairs)
        panels(output / "slim_intersection_compare.png", faces,
            [(suv, f"{name} / SLIM: {len(spairs)} illegal pairs", scounts),
             (final_uv, f"{name} / Ours: {len(pairs)} illegal pairs", counts)],
            cmap="YlOrRd", norm=Normalize(vmin=0, vmax=max(int(scounts.max()), int(counts.max()), 1)),
            colorbar="Illegal intersection pairs involving each face; shared scale")
    timeline = [{"global_step": r["global_step"], **r["extended_validity"]}
                for r in validity_checkpoints(initial_record, records, offsets)]
    result = {"dataset": name, "metric": "original-3D-area-weighted regularized SD", "dtype": "float64",
        "uv_scale": "native; no post-scaling", "initial_sd": initial_record["sd"], "final_sd": final_record["sd"],
        "original_validity": final_record["original_validity"], "extended_validity": final_record["extended_validity"],
        "export_validity": final_record["export_validity"], "geometry": final_record["geometry"],
        "network_composed": final_record["network_composed"], "slim": slim,
        "training_status": read(archive / "status.json"),
        "completion": read(audit / "completion.json") if (audit / "completion.json").exists()
                      else read(archive / "completion.json"),
        "archive": relative_path(archive, output), "source_commit": read(archive / "manifest.json")["head"],
        "input_sha256": read(archive / "manifest.json")["input_sha256"]}
    write_json(output / "summary.json", result)
    write_json(output / "config.json", config)
    write_json(output / "plot_metadata.json", {"native_float64": True, "sd_color_max": limits,
        "color_clipping_is_visual_only": True, "intersection_checkpoints": timeline,
        "intersection_marker": "circle; count, not failure", "area_color_max": area_upper})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "regularized_sd_f64", "strict_sd_f64", "flipped", "degenerate", "intersection_pairs"])
        for method, sd, val in [("ours", final_record["sd"], final_record["original_validity"])] + (
                [("slim", slim["sd"], slim["validity"])] if slim else []):
            writer.writerow([method, sd["regularized_f64"]["area_weighted_mean"], sd["strict_qr_f64"]["area_weighted_mean"],
                             val["flipped"], val["degenerate"], val["intersection_pairs"]])
    geometry_report = final_record["original_validity"]
    lines = [f"# {name} / v3.2 Results", "", f"Regularized area-weighted SD: {initial_record['sd']['regularized_f64']['area_weighted_mean']:.9g} -> {final_record['sd']['regularized_f64']['area_weighted_mean']:.9g}.",
        f"Final original mesh: {geometry_report['flipped']} flipped, {geometry_report['degenerate']} degenerate, {geometry_report['intersection_pairs']} illegal intersection pairs.",
        f"Network inverse status: {final_record['network_composed']['status']}; finite computation is not an accuracy certificate.", "",
        "![UV](uv_compare.png)", "![Distortion](distortion_compare.png)", "![Four metrics](four_metrics.png)", ""]
    if slim:
        lines.extend(["![SLIM comparison](slim_distortion_compare.png)", "![Intersections](slim_intersection_compare.png)", "",
            "SLIM and ours use the same float64 output metrics, but different strict/regularized optimization objectives and compute budgets.",
            "An overlapping SLIM output is not feasible under the global-injectivity requirement.", ""])
    lines.extend(["Heatmaps use explicitly clipped color scales, not clipped reported metrics. Comparison panels share a scale.",
        "Circles are intersection counts at audited steps, not failure markers. Intermediate steps are not independently intersection-tested.",
        "Geometric round trips are sampled, not continuous worst-case bounds. GEOS constructions use floating-point arithmetic.", "",
        f"[Run archive]({relative_path(archive, output)}) contains models, snapshots, raw metrics and logs. See run.json for the portable locator.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "completion.json", result["completion"])
    return result
