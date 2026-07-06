import torch


def identity_loss(uv: torch.Tensor, uv_ref: torch.Tensor) -> torch.Tensor:
    return (uv - uv_ref).pow(2).mean()
