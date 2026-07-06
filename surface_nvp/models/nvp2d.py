from __future__ import annotations

import torch
import torch.nn as nn

from .coupling2d import CouplingLayer2D


class NVP2D(nn.Module):
    def __init__(self, num_layers: int = 6, hidden_dim: int = 64, mlp_layers: int = 3, s_clamp: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer2D(i % 2, hidden_dim=hidden_dim, num_layers=mlp_layers, s_clamp=s_clamp)
            for i in range(num_layers)
        ])

    def forward(self, uv: torch.Tensor, return_logdet: bool = False):
        out = uv
        logdet = torch.zeros(uv.shape[0], dtype=uv.dtype, device=uv.device)
        for layer in self.layers:
            if return_logdet:
                out, ld = layer(out, return_logdet=True)
                logdet = logdet + ld
            else:
                out = layer(out)
        if return_logdet:
            return out, logdet
        return out

    def inverse(self, uv: torch.Tensor):
        out = uv
        for layer in reversed(self.layers):
            out = layer.inverse(out)
        return out
