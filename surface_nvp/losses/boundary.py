import torch


def boundary_position_loss(uv: torch.Tensor, uv_ref: torch.Tensor, boundary_indices: torch.Tensor) -> torch.Tensor:
    if boundary_indices.numel() == 0:
        return torch.zeros((), dtype=uv.dtype, device=uv.device)
    return (uv[boundary_indices] - uv_ref[boundary_indices]).pow(2).mean()
