from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from surface_nvp.io import MeshData, save_mesh
from research.mesh_pl_nvp.harmonic_local import HarmonicLocalHarmonicFlow
from research.mesh_pl_nvp.mesh_coupling import make_grid_triangulation
from research.mesh_pl_nvp.run_harmonic_local import observed_inverse
from research.mesh_pl_nvp.run_v31_audit import load_checkpoint
from research.mesh_pl_nvp.run_v32 import main, parse_args
from research.mesh_pl_nvp.tests.test_harmonic_local import _model


def test_v32_yaml_matches_cli_defaults():
    config = Path(__file__).resolve().parents[1]/"v3_2_default.yaml"
    built_in = parse_args(["--input", "mesh.obj", "--output-dir", "output"])
    loaded = parse_args(["--config", str(config), "--input", "mesh.obj", "--output-dir", "output"])
    assert {k: v for k, v in vars(built_in).items() if k != "config"} == {
        k: v for k, v in vars(loaded).items() if k != "config"}
    assert (loaded.harmonic_iters, loaded.local_iters, loaded.final_harmonic_iters) == (100, 500, 300)
    assert (loaded.harmonic_max_log_scale, loaded.harmonic_max_shift) == (.08, .20)
    assert (loaded.final_harmonic_max_log_scale, loaded.final_harmonic_max_shift) == (.16, .40)
    assert loaded.export_format == "obj"


def test_wide_h2_does_not_change_h1_or_local():
    old, scaffold = _model()
    wide = HarmonicLocalHarmonicFlow(scaffold.vertices_3d, scaffold.faces, scaffold.uv,
        old.first_harmonic.modes, old.first_harmonic.base_names, harmonic_cycles=1,
        local_cycles=1, local_hidden_dim=12, final_harmonic_max_log_scale=.16,
        final_harmonic_max_shift=.40).double()
    old.load_state_dict(wide.state_dict())
    assert wide.first_harmonic.max_log_scale == old.first_harmonic.max_log_scale == .08
    assert wide.final_harmonic.max_log_scale == .16
    with torch.no_grad():
        assert torch.equal(old.first_harmonic(), wide.first_harmonic())
        assert torch.equal(old.local(), wide.local())


@pytest.mark.parametrize("arguments", [["--snapshot-interval", "0"], ["--check-interval", "0"],
    ["--final-harmonic-max-shift", "nan"], ["--final-harmonic-max-log-scale", "-1"]])
def test_invalid_settings_rejected(arguments):
    with pytest.raises(ValueError):
        parse_args(["--input", "mesh.obj", "--output-dir", "output", *arguments])


def test_legacy_inverse_failure_is_explicit_and_oom_propagates():
    class Failing:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("inverse interval failed")
    assert observed_inverse(Failing(), None, None, include_final=True)["status"] == "failed"
    class Oom:
        def __call__(self, *args, **kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.OutOfMemoryError):
        observed_inverse(Oom(), None, None, include_final=True)


def test_full_entrypoint_preserves_results_when_inverse_fails(tmp_path, monkeypatch):
    uv, faces = make_grid_triangulation(5, 5)
    mesh = MeshData(torch.column_stack((uv, .04*torch.sin(uv[:, 0]))).numpy(), faces.numpy())
    input_path, output = tmp_path/"mesh.obj", tmp_path/"run"
    save_mesh(input_path, mesh)
    from research.mesh_pl_nvp.harmonic_modes import GlobalHarmonicModeFlow
    original = GlobalHarmonicModeFlow.forward
    def inverse_failure(self, *args, **kwargs):
        if kwargs.get("inverse"):
            raise RuntimeError("simulated inverse failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(GlobalHarmonicModeFlow, "forward", inverse_failure)
    arguments = ["--input", str(input_path), "--output-dir", str(output), "--device", "cpu",
        "--harmonic-iters", "1", "--local-iters", "2", "--final-harmonic-iters", "1",
        "--frequencies", "2", "--boundary-hat-counts", "--scaffold-rings", "2",
        "--harmonic-cycles", "1", "--local-cycles", "1", "--local-hidden-dim", "12"]
    main(arguments)
    completed = json.loads((output/"completion.json").read_text())
    assert completed["status"] == "complete"
    assert completed["network_non_ok_snapshots"]
    assert (output/"final.obj").exists()
    assert (output/"report"/"RESULTS.md").exists()
    model, payload = load_checkpoint(output/"final.model.pt")
    with torch.no_grad():
        assert torch.equal(model(), payload["uv"])
    with pytest.raises(FileExistsError):
        main(arguments)
    input_path.unlink()
    main(["audit", str(output), "--device", "cpu", "--audit-name", "portable", "--report-name", "portable_report"])
    assert (output/"portable"/"completion.json").exists()
    assert np.array_equal(payload["reference"]["vertices"], mesh.vertices)
