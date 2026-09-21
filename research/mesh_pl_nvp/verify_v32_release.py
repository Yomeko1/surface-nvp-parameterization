"""Verify release numerics against trusted, locally retained experiment states."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from . import run_harmonic_local as baseline
from .audit_metrics import sd_metrics
from .run_v31_audit import load_model, prepare, write_json
from .run_v32 import parse_args


def read_state(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def apply_parameters(model, snapshot):
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(snapshot["parameters"][key])


def verify(workspace, output, device):
    output.mkdir(parents=True, exist_ok=False)
    experiments = [
        ("Cow", workspace/"data/input/Cow/Cow_dABF.usda",
         workspace/"data/output/v3_1_cow_validation/20260920_full_01/original_affine",
         workspace/"data/output/v3_1_cow_validation/20260920_full_01/h2_affine_wide"),
        ("00027", workspace/"data/input/00027/Input.obj",
         workspace/"data/output/v3_1_audit_00027/20260917_full_01",
         workspace/"data/output/v3_1_followup_00027/20260920_full_01/h2_affine_wide"),
    ]
    results = []
    for name, input_path, source, adopted in experiments:
        args = parse_args(["--input", str(input_path), "--output-dir", str(output/name), "--device", device])
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        model, bundle, _, _ = prepare(args)
        expected_bundle = read_state(adopted/"reference.pt")
        tensors = ("vertices", "faces", "reference", "extended_vertices", "extended_faces", "modes", "boundary")
        reference_equal = {key: torch.equal(bundle[key], expected_bundle[key]) for key in tensors}
        final = read_state(adopted/"snapshots/final_harmonic_0300.pt")
        apply_parameters(model, final)
        with torch.no_grad():
            reproduced = model()
        exact = torch.equal(reproduced.cpu(), final["uv"])
        metrics, _ = sd_metrics(bundle["vertices"].numpy(), bundle["faces"].numpy(),
                                reproduced.cpu().numpy()[:len(bundle["vertices"])])
        comparisons = []
        for phase, begin, end in (("first_harmonic", 0, 50 if name == "Cow" else 10),
                                  ("local", 450 if name == "Cow" else 490, 500),
                                  ("final_harmonic", 250, 300)):
            directory = adopted if phase == "final_harmonic" else source
            snapshot = read_state(directory/"snapshots"/f"{phase}_{begin:04d}.pt")
            expected = read_state(directory/"snapshots"/f"{phase}_{end:04d}.pt")
            if phase == "local" and snapshot.get("schedule") is None:
                verified = workspace/"data/output/v3_1_followup_00027/20260920_full_01/continuation_verification/local"
                restored = read_state(verified/"snapshots"/f"{phase}_{begin:04d}.pt")
                if not torch.equal(restored["uv"], snapshot["uv"]) or not all(
                        torch.equal(value, restored["parameters"][key]) for key, value in snapshot["parameters"].items()):
                    raise ValueError("reconstructed schedule checkpoint differs from historical state")
                snapshot = restored
            apply_parameters(model, snapshot)
            model.set_trainable_stage(phase)
            base_uv = torch.load(directory/f"{phase}_input.pt", weights_only=True, map_location=device)
            positional = (base_uv, bundle["vertices"].to(device), bundle["faces"].to(device),
                          model.initial_uv, bundle["boundary"].to(device))
            common = dict(iterations=end, check_interval=50, gradient_clip=args.gradient_clip,
                          iteration_offset=0, resume_state=snapshot)
            if phase == "local":
                kwargs = {key.removeprefix("local_"): value for key, value in vars(args).items()
                          if key.startswith("local_risk_")}
                kwargs.update(dynamic_risk_threshold=args.local_dynamic_risk_threshold,
                              dynamic_floor_lr=args.local_dynamic_floor_lr)
                if snapshot.get("schedule") is None:
                    raise ValueError("historical local snapshot lacks recovery state")
                result = baseline._train_local_stage(model.local, *positional,
                    learning_rate=args.local_lr, **common, **kwargs)[0]
            else:
                result = baseline._train_harmonic_stage(getattr(model, phase), *positional,
                    learning_rate=args.harmonic_lr, phase=phase, **common)[0]
            row = {"phase": phase, "start": begin, "end": end,
                "uv_bitwise_equal": torch.equal(result.cpu(), expected["uv"]),
                "parameters_bitwise_equal": all(torch.equal(value.cpu(), expected["parameters"][key])
                    for key, value in getattr(model, phase).named_parameters(prefix=phase)),
                "max_abs_uv_difference": float((result.cpu()-expected["uv"]).abs().max())}
            comparisons.append(row)
            write_json(output/f"{name}_progress.json", comparisons)
        result = {"mesh": name, "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "reference_bitwise_equal": reference_equal, "final_forward_bitwise_equal": exact,
            "final_sd": metrics, "continuations": comparisons}
        results.append(result)
        write_json(output/"verification.json", results)
        if not all(reference_equal.values()) or not exact or not all(
                row["uv_bitwise_equal"] and row["parameters_bitwise_equal"] for row in comparisons):
            raise RuntimeError(f"release trajectory mismatch on {name}")
        print(f"verified {name}: SD={metrics['regularized_f64']['area_weighted_mean']:.9f}", flush=True)
        del model, reproduced, result, base_uv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    verify(args.workspace.resolve(), args.output.resolve(), args.device)
