from __future__ import annotations

import torch

from research.mesh_pl_nvp.mesh_coupling import (
    boundary_vertices,
    diagnose_embedding,
    make_grid_triangulation,
)
from research.mesh_pl_nvp.harmonic_modes import (
    build_piecewise_linear_boundary_harmonic_modes,
    build_radial_fourier_harmonic_modes,
)
from research.mesh_pl_nvp.multiring_scaffold import (
    build_multiring_outer_scaffold,
    multiring_transition_profile,
)


def test_multiring_scaffold_is_legal_and_only_last_ring_is_boundary() -> None:
    uv, faces = make_grid_triangulation(6, 6)
    vertices = torch.column_stack((uv, 0.1 * uv[:, 0] * uv[:, 1]))
    source_count = int(torch.count_nonzero(boundary_vertices(faces, uv.shape[0])))
    scaffold = build_multiring_outer_scaffold(
        vertices,
        faces,
        uv,
        scale=1.25,
        rings=4,
        transition_exponent=2.0,
    )
    assert scaffold.uv.shape[0] == uv.shape[0] + 4 * source_count
    extended_boundary = boundary_vertices(scaffold.faces, scaffold.uv.shape[0])
    assert torch.all(~extended_boundary[: uv.shape[0]])
    assert torch.all(extended_boundary[scaffold.outer_boundary])
    assert int(torch.count_nonzero(extended_boundary)) == source_count
    diagnostics = diagnose_embedding(scaffold.uv, scaffold.faces)
    assert diagnostics.flipped_faces == 0
    assert diagnostics.proper_edge_intersections == 0


def test_geometry_profile_makes_modes_follow_scaffold_interpolation() -> None:
    uv, faces = make_grid_triangulation(6, 6)
    vertices = torch.column_stack((uv, 0.1 * uv[:, 0] * uv[:, 1]))
    scaffold = build_multiring_outer_scaffold(
        vertices,
        faces,
        uv,
        scale=1.25,
        rings=4,
        transition_exponent=3.0,
    )
    transition_rings, transition_weights = multiring_transition_profile(
        scaffold,
        rings=4,
        transition_exponent=3.0,
    )
    builders = (
        lambda: build_radial_fourier_harmonic_modes(
            scaffold.uv,
            scaffold.faces,
            scaffold.original_boundary,
            scaffold.outer_boundary,
            frequencies=(2,),
            transition_rings=transition_rings,
            transition_weights=transition_weights,
        ),
        lambda: build_piecewise_linear_boundary_harmonic_modes(
            scaffold.uv,
            scaffold.faces,
            scaffold.original_boundary,
            scaffold.outer_boundary,
            control_counts=(4,),
            transition_rings=transition_rings,
            transition_weights=transition_weights,
        ),
    )
    for builder in builders:
        _, modes = builder()
        source_values = modes[:, scaffold.original_boundary]
        for ring, weight in zip(
            transition_rings, transition_weights, strict=True
        ):
            assert torch.allclose(modes[:, ring], weight * source_values)
        assert torch.equal(
            modes[:, scaffold.outer_boundary],
            torch.zeros_like(modes[:, scaffold.outer_boundary]),
        )
