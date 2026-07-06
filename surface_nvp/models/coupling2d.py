from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP


class CouplingLayer2D(nn.Module):
    def __init__(self, transform_index: int, hidden_dim: int = 64, num_layers: int = 3, s_clamp: float = 2.0):
        super().__init__()
        if transform_index not in (0, 1):
            raise ValueError("transform_index must be 0 or 1")
        self.transform_index = transform_index
        self.condition_index = 1 - transform_index
        self.s_clamp = s_clamp
        self.st = MLP(1, 2, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, uv: torch.Tensor, return_logdet: bool = False):
        cond = uv[:, self.condition_index:self.condition_index + 1]
        s, t = torch.chunk(self.st(cond), 2, dim=-1)
        s = torch.clamp(s, -self.s_clamp, self.s_clamp)
        out = uv.clone()
        out[:, self.transform_index:self.transform_index + 1] = (
            uv[:, self.transform_index:self.transform_index + 1] - t
        ) * torch.exp(-s)
        if return_logdet:
            return out, -s.squeeze(-1)
        return out

    def inverse(self, uv: torch.Tensor):
        cond = uv[:, self.condition_index:self.condition_index + 1]
        s, t = torch.chunk(self.st(cond), 2, dim=-1)
        s = torch.clamp(s, -self.s_clamp, self.s_clamp)
        out = uv.clone()
        out[:, self.transform_index:self.transform_index + 1] = (
            uv[:, self.transform_index:self.transform_index + 1] * torch.exp(s) + t
        )
        return out
