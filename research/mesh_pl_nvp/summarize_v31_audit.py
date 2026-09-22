"""Summarize the four agreed metrics without acceptance thresholds or selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run_v31_audit import archive_sources, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    return "not available" if value is None else f"{value:.9g}"


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validity_checkpoints(initial, records, offsets):
    points = [{"global_step": 0, **initial}]
    points.extend({**r, "global_step": offsets[r["phase"]]+r["step"]} for r in records)
    return [p for p in points if p["extended_validity"]["intersection_pairs"] is not None]


def summarize(output, audit_name="audit", report_name="report", source_store_dir=None):
    output = Path(output)
    if Path(report_name).name != report_name or report_name in {".", ".."}:
        raise ValueError("report name must be a simple subdirectory name")
    destination = output/report_name
    destination.mkdir(exist_ok=False)
    archive_sources(destination/"source.zip", source_store_dir)
    audit = output/audit_name
    manifest, status = read(output/"manifest.json"), read(output/"status.json")
    records, initial = read(audit/"summary.json"), read(audit/"initial.json")
    config = manifest["config"]
    offsets = {"first_harmonic": 0, "local": config["harmonic_iters"],
               "final_harmonic": config["harmonic_iters"]+config["local_iters"]}
    phase_lengths = {"first_harmonic": config["harmonic_iters"], "local": config["local_iters"],
                     "final_harmonic": config["final_harmonic_iters"]}
    training = [json.loads(line) for line in (output/"training.jsonl").read_text().splitlines()]
    training_index = {(r["phase"], r["step"]): r for r in training}
    points = [{"phase": "initial", "step": 0, "global_step": 0, **initial}]
    rows = []
    for r in records:
        t = offsets[r["phase"]]+r["step"]
        rows.append({"phase": r["phase"], "step": r["step"], "global_step": t,
            "sd_training_cuda_f64": training_index[r["phase"], r["step"]]["sd"],
            "sd_recomputed_cpu_f64": r["sd"]["regularized_f64"]["area_weighted_mean"],
            "sd_strict_qr_f64": r["sd"]["strict_qr_f64"]["area_weighted_mean"],
            "geometry_relative_max": nested(r, "geometry", "located", "relative", "max"),
            "network_relative_max_all": nested(r, "network_composed", "groups", "all", "relative", "max"),
            "network_status": r["network_composed"]["status"],
            "original_flipped": r["original_validity"]["flipped"],
            "extended_flipped": r["extended_validity"]["flipped"],
            "original_intersection_pairs": r["original_validity"]["intersection_pairs"],
            "extended_intersection_pairs": r["extended_validity"]["intersection_pairs"],
            "geometry_missing": r["geometry"]["missing"]})
        if r["step"] == phase_lengths[r["phase"]]:
            points.append({**r, "global_step": t})
    write_json(destination/"metrics_table.json", rows)
    consistency = {
        "training_status": status,
        "audited_snapshots": len(records),
        "training_rows": len(training),
        "gradient_rows": len((output/"gradients.jsonl").read_text().splitlines()) if (output/"gradients.jsonl").exists() else 0,
        "max_sd_cpu_vs_cuda_absolute": max(abs(r["sd_training_cuda_f64"]-r["sd_recomputed_cpu_f64"]) for r in rows),
        "network_failed_snapshots": [r["snapshot"] for r in records if r["network_composed"]["status"] != "ok"],
        "max_composed_recomputed_vs_cached": max((r.get("composed_recomputed_vs_cached_max_abs", 0.) for r in records)),
        "geometry_missing_total_over_snapshots": sum(r["geometry"]["missing"] for r in records),
        "scales": read(audit/"scales.json"),
    }
    write_json(destination/"consistency.json", consistency)
    x = np.array([r["global_step"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    tx = [offsets[r["phase"]]+r["step"] for r in training]
    axes[0, 0].plot(tx, [r["sd"] for r in training], color="#177a57", label="Training regularized SD")
    axes[0, 0].plot(x, [r["sd_strict_qr_f64"] for r in rows], color="#af5260", label="Strict SD (audit)")
    axes[0, 0].set(title="Original 3D-area-weighted SD", ylabel="SD")
    for key, color in (("max", "#177a57"), ("rms", "#b25c30"), ("p99", "#654c94")):
        values = [nested(r, "geometry", "located", "relative", key) for r in records]
        axes[0, 1].plot(x, values, color=color, label=key)
    axes[0, 1].set(title="Geometric round trip: sampled 3D -> UV -> 3D", ylabel="Distance / D3D", yscale="log")
    for key, color in (("original", "#177a57"), ("auxiliary", "#b25c30"), ("all", "#654c94")):
        values = [nested(r, "network_composed", "groups", key, "relative", "max") for r in records]
        axes[1, 0].plot(x, values, color=color, label=key)
    failed_x = [offsets[r["phase"]]+r["step"] for r in records if r["network_composed"]["status"] != "ok"]
    if failed_x:
        axes[1, 0].scatter(failed_x, np.full(len(failed_x), .96),
            transform=axes[1, 0].get_xaxis_transform(), marker="x", color="#bb3434", s=18,
            label="Inverse failed (status markers)")
    axes[1, 0].set(title="Network round trip: maximum vertex L2", ylabel="Distance / LUV0", yscale="log")
    finite_network = [nested(r, "network_composed", "groups", "all", "relative", "max") for r in records]
    if not any(value is not None and np.isfinite(value) and value > 0 for value in finite_network):
        axes[1, 0].set(yscale="linear", ylim=(0., 1.))
        axes[1, 0].set_yticks([])
        axes[1, 0].text(.5, .5, "No positive finite residuals; see inverse statuses",
            ha="center", va="center", transform=axes[1, 0].transAxes, fontsize=9)
    axes[1, 1].plot(x, [r["extended_flipped"] for r in rows], color="#b25c30", label="Extended flipped faces")
    checked = validity_checkpoints(initial, records, offsets)
    axes[1, 1].scatter([p["global_step"] for p in checked],
        [p["extended_validity"]["intersection_pairs"] for p in checked],
        color="#177a57", marker="o", s=35, facecolors="none", zorder=3,
        label="Intersection pairs at audited steps")
    from matplotlib.ticker import MaxNLocator
    max_count = max([r["extended_flipped"] for r in rows] + [p["extended_validity"]["intersection_pairs"] for p in checked])
    axes[1, 1].set(title="Validity counts", ylabel="Count", ylim=(-.1, max(1., max_count*1.1)))
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    if max_count == 0:
        axes[1, 1].text(.5, .5, "0 flipped faces; 0 intersection pairs at all checked steps",
            transform=axes[1, 1].transAxes, ha="center", va="center", fontsize=9, wrap=True)
    for ax in axes.flat:
        ax.set_xlabel("Completed parameter updates")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        for boundary in (config["harmonic_iters"], config["harmonic_iters"]+config["local_iters"]):
            ax.axvline(boundary, color="#777777", alpha=.4, linewidth=.7)
    fig.savefig(destination/"four_metrics.png", dpi=160)
    plt.close(fig)
    input_path = Path(config["input"])
    mesh_name = input_path.parent.name.rstrip("#") if input_path.stem == "Input" else input_path.stem
    release = manifest.get("release", "v3.1-audit")
    lines = [f"# {release} / {mesh_name} Audit Results", "", "No error acceptance thresholds or best-checkpoint selection.", "",
        f"Training status: {status['status']}. Updates: {status.get('updates', 'incomplete')}.",
        f"Base: {manifest['base_commit']}; branch: {manifest['branch']}.",
        f"D3D={consistency['scales']['D3D']:.16g}; LUV0={consistency['scales']['LUV0']:.16g}.", "",
        "| Stage | Regularized SD (CPU f64) | Geometry max / D3D | Network max / LUV0 (all) | Original flips / intersections | Extended flips / intersections |",
        "|---|---:|---:|---:|---:|---:|"]
    for p in points:
        o, e = p['original_validity'], p['extended_validity']
        network = p.get('network_composed')
        network_text = ("not applicable" if network is None else
            number(nested(network, 'groups', 'all', 'relative', 'max'))
            if network['status'] == 'ok' else "INVERSE FAILED: "+network['status'])
        values = [p['phase'], number(p['sd']['regularized_f64']['area_weighted_mean']),
            number(nested(p, 'geometry', 'located', 'relative', 'max')),
            network_text,
            f"{o['flipped']} / {o['intersection_pairs']}", f"{e['flipped']} / {e['intersection_pairs']}"]
        lines.append('| '+' | '.join(values)+' |')
    lines.extend(["", "![Four metrics](four_metrics.png)", "", "## Interpretation Limits", "",
        "- Geometry uses three fixed interior samples per original face; it is not a continuous worst-case bound.",
        "- Network residual is a state-space numerical cycle, not the geometric inverse and not undoing Adam.",
        "- Only original 3D faces contribute to SD. Regularized and strict formulas remain separate.",
        "- Circles mark intersection counts at every audited checkpoint, including H2 milestones; they are not failure markers.",
        "- Global intersections were checked at scheduled checkpoints, not at every update.",
        "- Intermediate null intersection counts mean not checked, never zero.",
        "- The GEOS intersection audit uses floating-point constructions, not a universal exact-arithmetic certificate.",
        "- Native f64, selected-format export/decode and float32 query variants are separate in each endpoint JSON.",
        "- CPU audit SD may differ slightly from the CUDA training reduction; consistency.json quantifies it.",
        "- Failed or nonfinite network audits remain explicit; no finite-only filtering is used.",
        "- Red crosses at the top of the network plot are failure statuses, not numerical residual values.",
        "- Connecting sampled metrics does not certify values between snapshots; no 1e-8 acceptance line is applied.",
        "", "## Reproducibility", "",
        "See manifest.json, reference.pt, training.jsonl, gradients.jsonl and snapshots/ in the run archive.",
        "Source snapshots are stored as source.zip or referenced by source.ref.json in the shared content-addressed store.", ""])
    (destination/"RESULTS.md").write_text('\n'.join(lines), encoding="utf-8")
    print(json.dumps(consistency, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--audit-name", default="audit")
    parser.add_argument("--report-name", default="report")
    args = parser.parse_args()
    summarize(args.output, args.audit_name, args.report_name)


if __name__ == "__main__":
    main()
