import numpy as np
import torch
import pytest

import surface_nvp.training.trainer as trainer_module

from surface_nvp.losses.distortion import symmetric_dirichlet_loss
from surface_nvp.training.supervised import fit_nvp_to_target
from surface_nvp.training.trainer import train_direct_uv, train_nvp
from surface_nvp.training.summary import build_run_summary
from surface_nvp.injectivity.validators import validate_uv, validate_uv_torch


def _square_mesh():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = vertices[:, :2].copy()
    return vertices, faces, uv


def test_direct_and_nvp_share_training_metrics():
    vertices, faces, uv0 = _square_mesh()
    _, direct_history, _ = train_direct_uv(
        vertices, faces, uv0, iters=1, lr=1e-4, check_interval=1, device="cpu"
    )
    _, _, nvp_history, _ = train_nvp(
        vertices,
        faces,
        uv0,
        iters=1,
        lr=1e-4,
        check_interval=1,
        device="cpu",
        return_info=True,
    )

    expected = {
        "loss",
        "loss_distortion",
        "loss_identity",
        "loss_jacobian",
        "weighted_loss_jacobian",
    }
    assert expected <= direct_history[0].keys()
    assert expected <= nvp_history[0].keys()
    assert direct_history[0]["is_valid"]
    assert nvp_history[0]["is_valid"]


def test_summary_accepts_v1_metrics_without_percentiles():
    distortion = {
        "symmetric_dirichlet_mean": 4.0,
        "symmetric_dirichlet_max": 5.0,
        "angle_distortion_mean_deg": 1.0,
        "angle_distortion_max_deg": 2.0,
    }
    payload = {
        "initial": {"distortion": distortion},
        "final": {
            "injectivity": {"is_valid": True, "num_flipped": 0, "num_intersections": 0, "min_signed_area": 0.1},
            "distortion": distortion,
        },
        "history": [],
    }

    summary = build_run_summary("old", 1, payload)

    assert summary["symmetric_dirichlet_p95"] is None


def test_symmetric_dirichlet_penalizes_collapsed_uv():
    vertices, faces, uv = _square_mesh()
    v_t = torch.as_tensor(vertices, dtype=torch.float32)
    f_t = torch.as_tensor(faces, dtype=torch.long)
    valid_uv = torch.as_tensor(uv, dtype=torch.float32)
    collapsed_uv = torch.zeros_like(valid_uv, requires_grad=True)

    valid_loss = symmetric_dirichlet_loss(v_t, f_t, valid_uv)
    collapsed_loss = symmetric_dirichlet_loss(v_t, f_t, collapsed_uv)

    assert torch.isfinite(collapsed_loss)
    assert collapsed_loss > valid_loss * 1e6
    collapsed_loss.backward()
    assert torch.isfinite(collapsed_uv.grad).all()
    assert collapsed_uv.grad.abs().sum() > 0.0


def test_nonfinite_uv_is_invalid_for_numpy_and_torch():
    _, faces, uv = _square_mesh()
    uv[0, 0] = np.nan

    numpy_metrics = validate_uv(uv, faces)
    torch_metrics = validate_uv_torch(torch.as_tensor(uv), torch.as_tensor(faces))

    assert not numpy_metrics["is_valid"]
    assert not torch_metrics["is_valid"]
    assert numpy_metrics["num_nonfinite"] == 1
    assert torch_metrics["num_nonfinite"] == 1


def test_training_options_are_validated():
    vertices, faces, uv0 = _square_mesh()
    with pytest.raises(ValueError, match="check_interval"):
        train_direct_uv(vertices, faces, uv0, iters=1, check_interval=0)
    with pytest.raises(ValueError, match="finite"):
        train_direct_uv(vertices, faces, uv0, iters=1, area_weight=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        train_direct_uv(vertices, faces, uv0, iters=1, seed=1.5)
    with pytest.raises(ValueError, match="lr_decay"):
        train_direct_uv(vertices, faces, uv0, iters=1, lr_decay=1.0)
    with pytest.raises(ValueError, match="removed in v2.1"):
        train_direct_uv(vertices, faces, uv0, iters=1, boundary_weight=10.0)


def test_nvp_seed_is_reproducible():
    vertices, faces, uv0 = _square_mesh()
    first, first_model, first_history = train_nvp(vertices, faces, uv0, iters=3, check_interval=1, seed=7)
    second, second_model, second_history = train_nvp(vertices, faces, uv0, iters=3, check_interval=1, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first_history == second_history
    for key, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[key], rtol=0.0, atol=0.0)


def test_spline_training_does_not_rescale_initial_uv_implicitly():
    vertices, faces, uv0 = _square_mesh()
    uv0 = uv0 * 0.1

    fitted, _, _, info = train_nvp(
        vertices,
        faces,
        uv0,
        coupling_type="spline",
        iters=1,
        lr=1e-12,
        min_lr=1e-15,
        check_interval=1,
        return_info=True,
    )

    assert info["selected_iteration"] == 0
    np.testing.assert_allclose(fitted, uv0, rtol=0.0, atol=2e-9)


def test_supervised_nvp_fits_identity_target():
    _, _, uv0 = _square_mesh()

    fitted, _, history, info = fit_nvp_to_target(
        uv0, uv0, coupling_type="affine", iters=1, check_interval=1
    )

    np.testing.assert_allclose(fitted, uv0, atol=1e-7)
    assert history[0]["mse"] == pytest.approx(0.0)
    assert info["selected_iteration"] == 1


def test_plateau_restart_reduces_learning_rate():
    vertices, faces, uv0 = _square_mesh()

    _, history, info = train_direct_uv(
        vertices,
        faces,
        uv0,
        iters=2,
        lr=1e-12,
        min_lr=1e-15,
        check_interval=1,
        plateau_patience=1,
    )

    assert info["plateau_restarts"] >= 1
    assert info["final_learning_rate"] < 1e-12
    assert any(entry.get("restarted_from_best") for entry in history)


def test_invalid_rollbacks_decay_learning_rate_cumulatively(monkeypatch):
    vertices, faces, uv0 = _square_mesh()

    def invalid_metrics(*_args, **_kwargs):
        return {
            "num_flipped": 1,
            "min_signed_area": -1.0,
            "num_nonfinite": 0,
            "num_intersections": 0,
            "intersections": [],
            "is_valid": False,
        }

    monkeypatch.setattr(trainer_module, "compute_metrics_torch", invalid_metrics)
    _, history, info = train_direct_uv(
        vertices,
        faces,
        uv0,
        iters=2,
        lr=1e-3,
        min_lr=1e-8,
        check_interval=1,
        lr_decay=0.5,
    )

    assert [entry["learning_rate"] for entry in history] == pytest.approx([5e-4, 2.5e-4])
    assert info["invalid_rollbacks"] == 2


def test_cosine_schedule_and_lbfgs_phase_are_recorded():
    vertices, faces, uv0 = _square_mesh()

    _, _, history, info = train_nvp(
        vertices,
        faces,
        uv0,
        iters=2,
        lr=1e-4,
        min_lr=1e-6,
        lr_schedule="cosine",
        lbfgs_iters=1,
        lbfgs_lr=0.1,
        lbfgs_check_interval=1,
        check_interval=1,
        mixing_type="rotation",
        return_info=True,
    )

    assert history[0]["phase"] == "adam"
    assert history[1]["learning_rate"] == pytest.approx(1e-6)
    assert history[-1]["phase"] == "lbfgs"
    assert info["lr_schedule"] == "cosine"
    assert info["lbfgs_iters"] == 1
