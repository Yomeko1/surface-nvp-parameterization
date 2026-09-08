from __future__ import annotations

import torch

from research.mesh_pl_nvp.harmonic_local import HarmonicLocalHarmonicFlow
from research.mesh_pl_nvp.harmonic_modes import build_radial_fourier_harmonic_modes
from research.mesh_pl_nvp.mesh_coupling import diagnose_embedding, make_grid_triangulation
from research.mesh_pl_nvp.scaffold import build_outer_scaffold


def _model() -> tuple[HarmonicLocalHarmonicFlow, object]:
    uv, faces = make_grid_triangulation(7, 7)
    xyz = torch.column_stack((uv, 0.08 * torch.sin(uv[:, 0] - uv[:, 1])))
    scaffold = build_outer_scaffold(xyz, faces, uv, scale=1.25)
    names, modes = build_radial_fourier_harmonic_modes(
        scaffold.uv,
        scaffold.faces,
        scaffold.original_boundary,
        scaffold.outer_boundary,
        frequencies=(2, 3),
    )
    model = HarmonicLocalHarmonicFlow(
        scaffold.vertices_3d,
        scaffold.faces,
        scaffold.uv,
        modes,
        names,
        harmonic_cycles=1,
        local_cycles=1,
        local_hidden_dim=12,
    ).double()
    return model, scaffold


def test_two_and_three_stage_compositions_are_explicitly_invertible() -> None:
    torch.manual_seed(47)
    model, scaffold = _model()
    with torch.no_grad():
        model.first_harmonic.parameters_raw.add_(0.08 * torch.randn_like(model.first_harmonic.parameters_raw))
        model.final_harmonic.parameters_raw.add_(0.08 * torch.randn_like(model.final_harmonic.parameters_raw))
        for parameter in model.local.parameters():
            parameter.add_(0.003 * torch.randn_like(parameter))

    for include_final in (False, True):
        mapped = model(include_final_harmonic=include_final)
        diagnostic = diagnose_embedding(mapped, scaffold.faces)
        assert diagnostic.flipped_faces == 0
        assert diagnostic.proper_edge_intersections == 0
        restored = model(
            mapped,
            inverse=True,
            include_final_harmonic=include_final,
        )
        assert float(torch.max(torch.abs(restored - scaffold.uv)).detach()) < 2.0e-8


def test_stage_selection_freezes_every_other_module() -> None:
    model, _ = _model()
    model.set_trainable_stage("local")
    assert not any(parameter.requires_grad for parameter in model.first_harmonic.parameters())
    assert all(parameter.requires_grad for parameter in model.local.parameters())
    assert not any(parameter.requires_grad for parameter in model.final_harmonic.parameters())
