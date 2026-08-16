from __future__ import annotations

import math

import torch
import torch.nn as nn
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.splines.rational_quadratic import DEFAULT_MIN_DERIVATIVE

from .mlp import MLP


class _SplineConditioner(MLP):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, num_bins: int):
        super().__init__(in_dim, out_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.hidden_features = hidden_dim
        identity_derivative = math.log(math.expm1(1.0 - DEFAULT_MIN_DERIVATIVE))
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.bias[2 * num_bins :] = identity_derivative

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        return super().forward(x)


class SplineCouplingLayer2D(nn.Module):
    def __init__(
        self,
        transform_index: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_bins: int = 8,
        tail_bound: float = 1.1,
    ):
        super().__init__()
        if transform_index not in (0, 1):
            raise ValueError("transform_index must be 0 or 1")
        mask = torch.zeros(2, dtype=torch.bool)
        mask[transform_index] = True

        def create_conditioner(in_dim: int, out_dim: int) -> nn.Module:
            return _SplineConditioner(in_dim, out_dim, hidden_dim, num_layers, num_bins)

        self.transform = PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_conditioner,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
        )

    def forward(self, uv: torch.Tensor, return_logdet: bool = False):
        out, logdet = self.transform(uv)
        if return_logdet:
            return out, logdet
        return out

    def inverse(self, uv: torch.Tensor):
        out, _ = self.transform.inverse(uv)
        return out
