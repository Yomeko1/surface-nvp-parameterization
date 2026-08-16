from __future__ import annotations

import torch
import torch.nn as nn


class DirectUV(nn.Module):
    def __init__(self, uv0: torch.Tensor):
        super().__init__()
        self.uv = nn.Parameter(uv0.detach().clone())

    def forward(self, _uv0: torch.Tensor) -> torch.Tensor:
        return self.uv
