from __future__ import annotations

import torch

from research.mesh_pl_nvp.harmonic_modes import (
    GlobalHarmonicModeFlow,
    boundary_hat_support_diagnostics,
    build_piecewise_linear_boundary_harmonic_modes,
    build_radial_fourier_harmonic_modes,
    legal_mode_interval,
    mode_coordinate,
    select_supported_boundary_hat_counts,
)
from research.mesh_pl_nvp.mesh_coupling import (
    diagnose_embedding,
    make_grid_triangulation,
    signed_double_areas,
)
from research.mesh_pl_nvp.scaffold import build_outer_scaffold


def _lift(uv: torch.Tensor) -> torch.Tensor:
    return torch.column_stack((uv, 0.1 * torch.sin(uv[:, 0] + 2.0 * uv[:, 1])))


def _case() -> tuple[GlobalHarmonicModeFlow, object]:
    uv, faces = make_grid_triangulation(8, 8)
    scaffold = build_outer_scaffold(_lift(uv), faces, uv, scale=1.25)
    names, modes = build_radial_fourier_harmonic_modes(
        scaffold.uv,
        scaffold.faces,
        scaffold.original_boundary,
        scaffold.outer_boundary,
        frequencies=(2, 3),
    )
    model = GlobalHarmonicModeFlow(
        scaffold.uv,
        scaffold.faces,
        modes,
        names,
        cycles=2,
        max_shift=0.2,
    ).double()
    return model, scaffold


def test_harmonic_modes_prescribe_source_and_fix_outer_boundary() -> None:
    model, scaffold = _case()
    boundary_rms = torch.sqrt(
        torch.mean(
            torch.sum(model.modes[:, scaffold.original_boundary].square(), dim=-1),
            dim=-1,
        )
    )
    assert torch.allclose(boundary_rms, torch.ones_like(boundary_rms), atol=2.0e-12)
    assert torch.equal(
        model.modes[:, scaffold.outer_boundary],
        torch.zeros_like(model.modes[:, scaffold.outer_boundary]),
    )


def test_piecewise_linear_boundary_hats_are_local_and_fix_outer_boundary() -> None:
    _, scaffold = _case()
    names, modes = build_piecewise_linear_boundary_harmonic_modes(
        scaffold.uv,
        scaffold.faces,
        scaffold.original_boundary,
        scaffold.outer_boundary,
        control_counts=(4, 8),
    )
    assert len(names) == 2 * (4 + 8)
    assert modes.shape == (len(names), scaffold.uv.shape[0], 2)
    assert torch.equal(
        modes[:, scaffold.outer_boundary],
        torch.zeros_like(modes[:, scaffold.outer_boundary]),
    )
    boundary_rms = torch.sqrt(
        torch.mean(
            torch.sum(modes[:, scaffold.original_boundary].square(), dim=-1),
            dim=-1,
        )
    )
    assert torch.allclose(boundary_rms, torch.ones_like(boundary_rms), atol=2.0e-12)
    finest_x = modes[names.index("boundary_hat_m8_c0_x"), scaffold.original_boundary]
    nonzero = torch.linalg.vector_norm(finest_x, dim=-1) > 1.0e-12
    assert 0 < int(torch.count_nonzero(nonzero)) < scaffold.original_boundary.numel()
    assert torch.equal(finest_x[:, 1], torch.zeros_like(finest_x[:, 1]))


def test_boundary_hat_support_selection_detects_unsampled_controls() -> None:
    square = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    selected, diagnostics = select_supported_boundary_hat_counts(
        square,
        torch.arange(4),
        (4, 8),
    )
    assert selected == (4,)
    assert diagnostics[0]["supported"] is True
    assert diagnostics[0]["minimum_peak_weight"] == 1.0
    assert diagnostics[1]["supported"] is False
    assert diagnostics[1]["reason"] == "more_controls_than_boundary_vertices"

    nonuniform = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=torch.float64,
    )
    diagnostics = boundary_hat_support_diagnostics(
        nonuniform,
        torch.arange(4),
        (4,),
        minimum_peak_weight=1.0e-12,
    )
    assert diagnostics[0]["supported"] is False
    assert diagnostics[0]["minimum_active_vertices"] == 0
    assert diagnostics[0]["unsupported_control_indices"] == [3]


def test_quadratic_legal_interval_is_invariant_along_mode_orbit() -> None:
    model, scaffold = _case()
    mode = model.modes[0]
    interval0 = legal_mode_interval(
        scaffold.uv, mode, scaffold.faces, model.area_floor
    )
    increment = 0.2 * torch.minimum(
        interval0.coefficient - interval0.lower,
        interval0.upper - interval0.coefficient,
    )
    moved = scaffold.uv + increment * mode
    interval1 = legal_mode_interval(moved, mode, scaffold.faces, model.area_floor)
    _, residual0 = mode_coordinate(scaffold.uv, mode)
    _, residual1 = mode_coordinate(moved, mode)

    assert torch.allclose(interval0.lower, interval1.lower, atol=2.0e-11, rtol=0.0)
    assert torch.allclose(interval0.upper, interval1.upper, atol=2.0e-11, rtol=0.0)
    assert torch.allclose(
        interval1.coefficient,
        interval0.coefficient + increment,
        atol=2.0e-12,
        rtol=0.0,
    )
    assert torch.allclose(residual0, residual1, atol=2.0e-12, rtol=0.0)


def test_global_harmonic_flow_is_legal_and_explicitly_invertible() -> None:
    torch.manual_seed(41)
    model, scaffold = _case()
    identity = model()
    assert torch.allclose(identity, scaffold.uv, atol=2.0e-12, rtol=0.0)
    with torch.no_grad():
        model.parameters_raw.add_(0.25 * torch.randn_like(model.parameters_raw))
    mapped, diagnostics = model(return_diagnostics=True)
    assert bool(torch.all(signed_double_areas(mapped, scaffold.faces) > model.area_floor))
    embedding = diagnose_embedding(mapped, scaffold.faces)
    assert embedding.flipped_faces == 0
    assert embedding.proper_edge_intersections == 0
    assert torch.equal(
        mapped[scaffold.outer_boundary], scaffold.uv[scaffold.outer_boundary]
    )
    assert float(diagnostics.q_values.detach().max()) < 1.0

    restored = model(mapped, inverse=True)
    assert float(torch.max(torch.abs(restored - scaffold.uv)).detach()) < 2.0e-10


def test_global_harmonic_parameters_receive_sd_like_gradient() -> None:
    model, _ = _case()
    mapped = model()
    weights = torch.tensor([1.0, 1.37], dtype=mapped.dtype)
    loss = torch.sum(mapped.square() * weights)
    loss.backward()
    assert model.parameters_raw.grad is not None
    assert float(model.parameters_raw.grad.square().sum()) > 0.0
