import torch

from surface_nvp.models import NVP2D


def test_nvp2d_positive_logdet():
    model = NVP2D(num_layers=4)
    uv = torch.randn(16, 2)
    _, logdet = model(uv, return_logdet=True)
    assert torch.isfinite(logdet).all()
