from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from research.mesh_pl_nvp.audit_metrics import (
    distance_statistics, geometric_roundtrip, network_residual, scales, sd_metrics, validity,
)
from research.mesh_pl_nvp.run_v31_audit import Recorder
from research.mesh_pl_nvp.run_harmonic_local import _train_harmonic_stage, _train_local_stage
from research.mesh_pl_nvp.tests.test_harmonic_local import _model


def square():
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    return np.column_stack((uv, np.zeros(4))), np.array([[0, 1, 2], [0, 2, 3]]), uv


def test_geometric_cycle_and_normalization():
    v, f, u = square()
    for factor in (1e-7, 1., 1e7):
        s = scales(v*factor, u*factor)
        result, _ = geometric_roundtrip(v*factor, f, u*factor, s['D3D'])
        assert result['missing'] == 0
        assert result['known_face']['relative']['max'] < 1e-14
        assert result['located']['relative']['max'] < 1e-14


def test_network_l2_and_legacy_coordinate_are_distinct():
    u = np.zeros((3, 2))
    restored = np.array([[3., 4.], [0., 0.], [6., 8.]])
    result, _ = network_residual(restored, u, 2, 5.)
    assert result['groups']['original']['absolute']['max'] == 5
    assert result['groups']['all']['relative']['max'] == 2
    assert result['legacy_max_abs_coordinate'] == 8
    assert distance_statistics([1., np.nan], 1)['absolute'] is None


def test_sd_matches_objective_and_strict_isometry():
    v, f, u = square()
    result, _ = sd_metrics(v, f, u)
    assert abs(result['strict_qr_f64']['area_weighted_mean']-4.) < 1e-12
    assert abs(result['regularized_f64']['area_weighted_mean']-(2+2/(1+1e-4)**2)) < 1e-12
    from surface_nvp.training.metrics import compute_distortion_metrics
    assert result['legacy_regularized_f32']['area_weighted_mean'] == compute_distortion_metrics(v, f, u)['symmetric_dirichlet_area_weighted_mean']


def test_positive_tiny_faces_not_flipped_and_shared_edges_not_intersections():
    _, f, u = square()
    for factor in (1e-8, 1., 1e8):
        result, _ = validity(u*factor, f)
        assert result['flipped'] == 0
        assert result['degenerate'] == 0
        assert result['intersection_pairs'] == 0
    assert validity(u*1e-8, f)[0]['legacy_num_flipped'] == 2
    assert validity(u, f, intersections=False)[0]['intersection_pairs'] is None


def test_overlap_detected_even_with_shared_vertex_or_edge():
    u = np.array([[0., 0.], [2., 0.], [0., 2.], [.5, .5], [.8, .5]])
    for f in (np.array([[0, 1, 2], [0, 3, 4]]), np.array([[0, 1, 2], [0, 1, 3]])):
        assert validity(u, f)[0]['intersection_pairs'] == 1
    _, f, u = square()
    f[1] = f[1, ::-1]
    assert validity(u, f)[0]['flipped'] == 1


def test_point_location_failure_not_hidden():
    v, f, u = square()
    result, _ = geometric_roundtrip(v, f, u, 1., decode_uv=u+100)
    assert result['missing'] == 6
    assert result['located']['status'] == 'nonfinite'
    assert result['located']['absolute'] is None


def test_bounded_edge_pairs_and_legacy_count_agree(monkeypatch):
    from research.mesh_pl_nvp import mesh_coupling as m
    bounded = m._upper_triangle_batches
    for n in (0, 1, 2, 9, 32):
        for batch in (1, 7, 16, 1000):
            chunks = list(bounded(n, batch, 'cpu'))
            pairs = [tuple(p) for chunk in chunks for p in chunk.T.tolist()]
            assert sorted(pairs) == [(i, j) for i in range(n) for j in range(i+1, n)]
            assert all(chunk.shape[1] <= batch for chunk in chunks)
    uv, faces = m.make_grid_triangulation(5, 5)
    generator = torch.Generator().manual_seed(911)
    uv = uv + .5*torch.randn(uv.shape, generator=generator, dtype=uv.dtype)
    actual = m.count_proper_edge_intersections(uv, faces, pair_batch_size=17)
    monkeypatch.setattr(m, '_upper_triangle_batches', lambda n, b, d: [torch.triu_indices(n, n, 1, device=d)])
    assert actual == m.count_proper_edge_intersections(uv, faces)


@pytest.mark.parametrize('spline', [False, True])
def test_observer_preserves_parameters_history_and_rng(tmp_path, spline):
    torch.manual_seed(81)
    model, scaffold = _model()
    if spline:
        from research.mesh_pl_nvp.harmonic_local import HarmonicLocalHarmonicFlow
        model = HarmonicLocalHarmonicFlow(scaffold.vertices_3d, scaffold.faces, scaffold.uv,
            model.first_harmonic.modes, model.first_harmonic.base_names,
            harmonic_cycles=1, local_cycles=1, local_hidden_dim=12,
            local_radial_map='atanh', local_latent_transform='spline').double()
    n = int(scaffold.original_vertex_count)
    common = dict(iterations=2, learning_rate=.001, check_interval=1, gradient_clip=10., iteration_offset=0)
    for phase in ('first_harmonic', 'local'):
        left, right = copy.deepcopy(model), copy.deepcopy(model)
        left.set_trainable_stage(phase)
        right.set_trainable_stage(phase)
        target = tmp_path/phase
        target.mkdir()
        (target/'snapshots').mkdir()
        recorder = Recorder(target, right, phase, 2, interval=1)
        args = (scaffold.uv, scaffold.vertices_3d[:n], scaffold.faces[:scaffold.original_face_count],
                scaffold.uv, scaffold.original_boundary)
        function = _train_local_stage if phase == 'local' else _train_harmonic_stage
        extra = ({'risk_adaptive': True, 'risk_recovery': True, 'risk_threshold': .01,
                  'risk_warmup_iters': 2} if phase == 'local' else {'phase': phase})
        rng = torch.get_rng_state().clone()
        a = function(getattr(left, phase), *args, **common, **extra)
        b = function(getattr(right, phase), *args, **common, **extra, observer=recorder)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(a[0], b[0])
        assert a[1] == b[1]
        if phase == 'local':
            assert a[3] == b[3]
            assert a[3]['safety_triggered']
        for p, q in zip(left.parameters(), right.parameters()):
            assert torch.equal(p, q)


def test_nonfinite_and_exact_degeneracy_are_not_real_flips():
    _, f, u = square()
    u[2] = u[1]
    result, _ = validity(u, f)
    assert result['degenerate'] == 1
    assert result['flipped'] == 0
    assert result['intersection_pairs'] is None
    u[0] = np.nan
    result, _ = validity(u, f)
    assert result['nonfinite_faces'] == 2
    assert result['flipped'] == 0


def test_oom_runtime_errors_are_classified():
    from research.mesh_pl_nvp.run_v31_audit import is_oom
    assert is_oom(RuntimeError('DefaultCPUAllocator: not enough memory'))
    assert is_oom(torch.OutOfMemoryError('CUDA out of memory'))
    assert not is_oom(RuntimeError('hard area assertion failed'))
