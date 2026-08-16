import torch

from surface_nvp.models import NVP2D


def test_nvp2d_positive_logdet():
    model = NVP2D(num_layers=4)
    uv = torch.randn(16, 2)
    _, logdet = model(uv, return_logdet=True)
    assert torch.isfinite(logdet).all()


def test_spline_nvp_identity_inverse_and_positive_jacobian():
    model = NVP2D(num_layers=4, coupling_type="spline", spline_bins=8)
    uv = torch.rand(12, 2) * 1.8 - 0.9
    model.set_domain(uv)

    mapped, logdet = model(uv, return_logdet=True)
    recovered = model.inverse(mapped)

    torch.testing.assert_close(mapped, uv, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered, uv, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(logdet).all()

    point = uv[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(lambda x: model(x.unsqueeze(0)).squeeze(0), point)
    assert torch.det(jacobian) > 0.0


def test_trained_spline_nvp_remains_invertible_and_orientation_preserving():
    model = NVP2D(num_layers=4, coupling_type="spline", spline_bins=8)
    uv = torch.rand(12, 2) * 1.8 - 0.9
    model.set_domain(uv)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 1e-2)

    mapped = model(uv)
    recovered = model.inverse(mapped)

    torch.testing.assert_close(recovered, uv, atol=2e-4, rtol=2e-4)
    point = uv[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(lambda x: model(x.unsqueeze(0)).squeeze(0), point)
    assert torch.det(jacobian) > 0.0
