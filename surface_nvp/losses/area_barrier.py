import torch

from surface_nvp.injectivity.signed_area import torch_signed_areas


def area_barrier_loss(uv: torch.Tensor, faces: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    areas = torch_signed_areas(uv, faces)
    return torch.relu(eps - areas).pow(2).mean()
