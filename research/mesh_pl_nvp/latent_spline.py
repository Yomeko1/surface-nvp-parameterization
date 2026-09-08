"""Conditional rational-quadratic spline coupling in polygon latent space."""

from __future__ import annotations

import math
from typing import Optional

import torch
from nflows.transforms.splines.rational_quadratic import (
    DEFAULT_MIN_DERIVATIVE,
    unconstrained_rational_quadratic_spline,
)
from torch import Tensor, nn


class LatentSplineConditioner(nn.Module):
    """One coordinate conditioner with an exact identity start."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_bins: int = 8,
        tail_bound: float = 8.0,
        transformed_index: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("spline num_bins must be at least 2")
        if tail_bound <= 0.0 or not math.isfinite(tail_bound):
            raise ValueError("spline tail_bound must be finite and positive")
        if transformed_index not in (0, 1):
            raise ValueError("transformed_index must be 0 or 1")
        parameter_count = 3 * num_bins - 1
        factory_kwargs = {"dtype": dtype, "device": device}
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, parameter_count, **factory_kwargs),
        )
        identity_derivative = math.log(
            math.expm1(1.0 - DEFAULT_MIN_DERIVATIVE)
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[2 * num_bins :].fill_(identity_derivative)
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.transformed_index = int(transformed_index)

    def spline_parameters(
        self,
        features: Tensor,
        retained_coordinate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        inputs = torch.cat((features, retained_coordinate[:, None]), dim=-1)
        raw = self.network(inputs)
        widths = raw[:, : self.num_bins]
        heights = raw[:, self.num_bins : 2 * self.num_bins]
        derivatives = raw[:, 2 * self.num_bins :]
        return widths, heights, derivatives


def batched_legal_axis_intervals(
    points: Tensor,
    A: Tensor,
    b: Tensor,
    mask: Tensor,
    *,
    transformed_index: int,
) -> tuple[Tensor, Tensor]:
    """Intersect polygon half-planes along one axis through each point."""
    if transformed_index not in (0, 1):
        raise ValueError("transformed_index must be 0 or 1")
    retained_index = 1 - transformed_index
    coefficients = A[..., transformed_index]
    rhs = b - A[..., retained_index] * points[:, retained_index, None]
    tolerance = 32.0 * torch.finfo(points.dtype).eps
    positive = mask & (coefficients > tolerance)
    negative = mask & (coefficients < -tolerance)
    safe_coefficients = torch.where(
        positive | negative, coefficients, torch.ones_like(coefficients)
    )
    intersections = rhs / safe_coefficients
    lower = torch.where(
        negative, intersections, torch.full_like(intersections, -torch.inf)
    ).amax(dim=-1)
    upper = torch.where(
        positive, intersections, torch.full_like(intersections, torch.inf)
    ).amin(dim=-1)
    if bool(torch.any(~torch.isfinite(lower) | ~torch.isfinite(upper)).detach()):
        raise ValueError("a polygon has an unbounded axis interval")
    current = points[:, transformed_index]
    if bool(torch.any((current <= lower) | (current >= upper)).detach()):
        minimum_clearance = float(
            torch.minimum(current - lower, upper - current).detach().amin()
        )
        raise ValueError(
            "a point is outside its legal axis interval: "
            f"minimum_clearance={minimum_clearance:.17g}"
        )
    return lower, upper


def interval_spline_coupling(
    values: Tensor,
    lower: Tensor,
    upper: Tensor,
    retained_coordinate: Tensor,
    features: Tensor,
    conditioner: LatentSplineConditioner,
    *,
    inverse: bool = False,
) -> Tensor:
    """Apply an RQS in the atanh chart of an exact legal scalar interval."""
    if values.ndim != 1:
        raise ValueError("values must have shape [batch]")
    if not (
        lower.shape
        == upper.shape
        == retained_coordinate.shape
        == values.shape
    ):
        raise ValueError("interval and retained-coordinate shapes must match values")
    if features.shape[0] != values.shape[0]:
        raise ValueError("features batch dimension must match values")
    width = upper - lower
    if bool(torch.any(~torch.isfinite(width)).detach()) or bool(
        torch.any(width <= 0.0).detach()
    ):
        raise ValueError("spline intervals must be finite and non-empty")
    normalized = 2.0 * (values - lower) / width - 1.0
    if bool(torch.any(torch.abs(normalized).detach() >= 1.0)):
        maximum = float(torch.abs(normalized).detach().amax())
        raise ValueError(
            f"a spline input lies outside its legal interval: max_abs={maximum:.17g}"
        )
    latent = torch.atanh(normalized)
    widths, heights, derivatives = conditioner.spline_parameters(
        features,
        retained_coordinate,
    )
    transformed, _ = unconstrained_rational_quadratic_spline(
        latent,
        widths,
        heights,
        derivatives,
        inverse=inverse,
        tails="linear",
        tail_bound=conditioner.tail_bound,
    )
    mapped_normalized = torch.tanh(transformed)
    return lower + 0.5 * (mapped_normalized + 1.0) * width
