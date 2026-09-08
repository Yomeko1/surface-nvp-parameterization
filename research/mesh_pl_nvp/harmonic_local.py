"""Composable global-harmonic/local-vertex/global-harmonic PL-NVP flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .batched_coupling import BatchedMeshCouplingFlow
from .harmonic_modes import (
    GlobalHarmonicModeFlow,
    HarmonicModeDiagnostics,
)
from .trainable_coupling import CouplingDiagnostics


@dataclass(frozen=True)
class HarmonicLocalDiagnostics:
    first_harmonic: HarmonicModeDiagnostics
    local: CouplingDiagnostics
    final_harmonic: HarmonicModeDiagnostics | None


class HarmonicLocalHarmonicFlow(nn.Module):
    """Certified composition ``H1 -> local mesh PL-NVP -> H2``."""

    def __init__(
        self,
        vertices_3d: Tensor,
        faces: Tensor,
        initial_uv: Tensor,
        modes: Tensor,
        names: Iterable[str],
        *,
        harmonic_cycles: int = 2,
        harmonic_max_log_scale: float = 0.08,
        harmonic_max_shift: float = 0.20,
        area_margin_ratio: float = 1.0e-6,
        local_cycles: int = 4,
        local_hidden_dim: int = 32,
        local_feature_set: str = "basic",
        local_radial_map: str = "softsign",
        local_max_log_scale: float = 0.08,
        local_max_shift_fraction: float = 0.04,
        local_center_iterations: int = 12,
        local_latent_transform: str = "affine",
        local_spline_bins: int = 8,
        local_spline_bound: float = 8.0,
    ) -> None:
        super().__init__()
        names = tuple(names)
        harmonic_arguments = {
            "cycles": harmonic_cycles,
            "max_log_scale": harmonic_max_log_scale,
            "max_shift": harmonic_max_shift,
            "area_margin_ratio": area_margin_ratio,
        }
        self.first_harmonic = GlobalHarmonicModeFlow(
            initial_uv, faces, modes, names, **harmonic_arguments
        )
        self.local = BatchedMeshCouplingFlow(
            vertices_3d,
            faces,
            initial_uv,
            cycles=local_cycles,
            hidden_dim=local_hidden_dim,
            feature_set=local_feature_set,
            radial_map=local_radial_map,
            max_log_scale=local_max_log_scale,
            max_shift_fraction=local_max_shift_fraction,
            center_iterations=local_center_iterations,
            latent_transform=local_latent_transform,
            spline_bins=local_spline_bins,
            spline_bound=local_spline_bound,
        )
        self.final_harmonic = GlobalHarmonicModeFlow(
            initial_uv, faces, modes, names, **harmonic_arguments
        )

    @property
    def initial_uv(self) -> Tensor:
        return self.first_harmonic.initial_uv

    @property
    def faces(self) -> Tensor:
        return self.first_harmonic.faces

    def set_trainable_stage(self, stage: str) -> None:
        if stage not in {"first_harmonic", "local", "final_harmonic", "none"}:
            raise ValueError(f"unknown trainable stage: {stage}")
        for name, module in (
            ("first_harmonic", self.first_harmonic),
            ("local", self.local),
            ("final_harmonic", self.final_harmonic),
        ):
            module.requires_grad_(name == stage)

    def forward(
        self,
        uv: Tensor | None = None,
        *,
        inverse: bool = False,
        include_final_harmonic: bool = True,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, HarmonicLocalDiagnostics]:
        result = self.initial_uv if uv is None else uv
        if inverse:
            final_diagnostics = None
            if include_final_harmonic:
                result, final_diagnostics = self.final_harmonic(
                    result, inverse=True, return_diagnostics=True
                )
            result, local_diagnostics = self.local(
                result, inverse=True, return_diagnostics=True
            )
            result, first_diagnostics = self.first_harmonic(
                result, inverse=True, return_diagnostics=True
            )
        else:
            result, first_diagnostics = self.first_harmonic(
                result, return_diagnostics=True
            )
            result, local_diagnostics = self.local(
                result, return_diagnostics=True
            )
            final_diagnostics = None
            if include_final_harmonic:
                result, final_diagnostics = self.final_harmonic(
                    result, return_diagnostics=True
                )
        if return_diagnostics:
            return result, HarmonicLocalDiagnostics(
                first_diagnostics,
                local_diagnostics,
                final_diagnostics,
            )
        return result
