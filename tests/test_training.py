import numpy as np
import torch
import pytest

from surface_nvp.losses.distortion import symmetric_dirichlet_loss
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
        "loss_boundary",
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
