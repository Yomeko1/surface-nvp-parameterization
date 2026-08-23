from __future__ import annotations

import torch
import torch.nn as nn


class RotationLayer2D(nn.Module):
    """Learnable orientation-preserving 2D mixing with zero log determinant."""

    def __init__(self, initial_angle: float = 0.0):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(float(initial_angle)))

    def _matrix(self) -> torch.Tensor:
        cosine = torch.cos(self.angle)
        sine = torch.sin(self.angle)
        return torch.stack(
            [
                torch.stack([cosine, -sine]),
                torch.stack([sine, cosine]),
            ]
        )

    def forward(self, uv: torch.Tensor, return_logdet: bool = False):
        out = uv @ self._matrix().transpose(0, 1)
        if return_logdet:
            logdet = torch.zeros(uv.shape[0], dtype=uv.dtype, device=uv.device)
            return out, logdet
        return out

    def inverse(self, uv: torch.Tensor) -> torch.Tensor:
        return uv @ self._matrix()
