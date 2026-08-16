from __future__ import annotations

import math
from numbers import Integral

import torch
import torch.nn as nn

from .coupling2d import CouplingLayer2D
from .spline_coupling2d import SplineCouplingLayer2D


class NVP2D(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = 64,
        mlp_layers: int = 3,
        s_clamp: float = 2.0,
        coupling_type: str = "affine",
        spline_bins: int = 8,
        spline_bound: float = 1.1,
    ):
        super().__init__()
        numeric_values = (num_layers, hidden_dim, mlp_layers, s_clamp, spline_bins, spline_bound)
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("model options must be finite")
        if not all(isinstance(value, Integral) for value in (num_layers, hidden_dim, mlp_layers, spline_bins)):
            raise ValueError("model dimensions and spline_bins must be integers")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if mlp_layers <= 0:
            raise ValueError("mlp_layers must be positive")
        if s_clamp <= 0.0:
            raise ValueError("s_clamp must be positive")
        if spline_bins < 2:
            raise ValueError("spline_bins must be at least 2")
        if spline_bound <= 0.0:
            raise ValueError("spline_bound must be positive")
        if coupling_type == "affine":
            make_layer = lambda i: CouplingLayer2D(
                i % 2, hidden_dim=hidden_dim, num_layers=mlp_layers, s_clamp=s_clamp
            )
        elif coupling_type == "spline":
            make_layer = lambda i: SplineCouplingLayer2D(
                i % 2,
                hidden_dim=hidden_dim,
                num_layers=mlp_layers,
                num_bins=spline_bins,
                tail_bound=spline_bound,
            )
        else:
            raise ValueError("coupling_type must be 'affine' or 'spline'")
        self.layers = nn.ModuleList([make_layer(i) for i in range(num_layers)])
        self.coupling_type = coupling_type
        self.spline_bound = spline_bound
        self.register_buffer("domain_center", torch.zeros(2))
        self.register_buffer("domain_scale", torch.ones(2))
        if coupling_type == "spline":
            self.global_log_scale = nn.Parameter(torch.zeros(2))
            self.global_translation = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("global_log_scale", None)
            self.register_parameter("global_translation", None)

    def set_domain(self, uv: torch.Tensor) -> None:
        center = 0.5 * (uv.amin(dim=0) + uv.amax(dim=0))
        half_extent = 0.5 * (uv.amax(dim=0) - uv.amin(dim=0))
        target_extent = 0.9 * self.spline_bound
        scale = (half_extent / target_extent).clamp_min(1e-8)
        with torch.no_grad():
            self.domain_center.copy_(center)
            self.domain_scale.copy_(scale)

    def forward(self, uv: torch.Tensor, return_logdet: bool = False):
        out = (uv - self.domain_center) / self.domain_scale
        logdet = torch.zeros(uv.shape[0], dtype=uv.dtype, device=uv.device)
        for layer in self.layers:
            if return_logdet:
                out, ld = layer(out, return_logdet=True)
                logdet = logdet + ld
            else:
                out = layer(out)
        out = out * self.domain_scale + self.domain_center
        if self.global_log_scale is not None:
            out = (
                (out - self.domain_center) * torch.exp(self.global_log_scale)
                + self.domain_center
                + self.global_translation
            )
            logdet = logdet + self.global_log_scale.sum()
        if return_logdet:
            return out, logdet
        return out

    def inverse(self, uv: torch.Tensor):
        out = uv
        if self.global_log_scale is not None:
            out = (
                (out - self.global_translation - self.domain_center)
                * torch.exp(-self.global_log_scale)
                + self.domain_center
            )
        out = (out - self.domain_center) / self.domain_scale
        for layer in reversed(self.layers):
            out = layer.inverse(out)
        return out * self.domain_scale + self.domain_center
