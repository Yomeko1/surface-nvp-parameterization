"""Certified global harmonic-mode coupling layers for mesh PL-NVP research.

The source boundary receives low-frequency radial Fourier displacements.  A
discrete harmonic solve extends each displacement through the original disk,
while the outer scaffold boundary remains fixed.  Moving the complete mesh
along one fixed mode reduces every signed triangle area to a scalar quadratic.
The connected legal coefficient interval can therefore be computed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import torch
from torch import Tensor, nn

from .mesh_coupling import signed_double_areas


@dataclass(frozen=True)
class ModeInterval:
    lower: Tensor
    upper: Tensor
    coefficient: Tensor


@dataclass(frozen=True)
class HarmonicModeDiagnostics:
    names: tuple[str, ...]
    coefficients_before: Tensor
    coefficients_after: Tensor
    lower_bounds: Tensor
    upper_bounds: Tensor
    q_values: Tensor
    minimum_double_areas: Tensor


def _vertex_adjacency(faces: np.ndarray, vertex_count: int) -> list[set[int]]:
    adjacency = [set() for _ in range(vertex_count)]
    for face in faces:
        a, b, c = map(int, face)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def harmonic_extend_boundary_values(
    faces: Tensor,
    vertex_count: int,
    constrained_vertices: Tensor,
    constrained_values: Tensor,
) -> Tensor:
    """Uniform-Laplacian extension for several vector-valued boundary modes.

    ``constrained_values`` has shape ``[mode_count, constrained_count, 2]``.
    The sparse solve is intentionally performed once on CPU because modes are
    fixed preprocessing data and do not participate in autograd.
    """
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    constrained_np = (
        constrained_vertices.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    values_np = constrained_values.detach().cpu().numpy().astype(np.float64, copy=False)
    if values_np.ndim != 3 or values_np.shape[1:] != (len(constrained_np), 2):
        raise ValueError("constrained_values must have shape [mode_count, boundary_count, 2]")

    is_constrained = np.zeros(vertex_count, dtype=bool)
    is_constrained[constrained_np] = True
    free = np.flatnonzero(~is_constrained)
    free_row = np.full(vertex_count, -1, dtype=np.int64)
    free_row[free] = np.arange(len(free), dtype=np.int64)
    constrained_row = np.full(vertex_count, -1, dtype=np.int64)
    constrained_row[constrained_np] = np.arange(len(constrained_np), dtype=np.int64)

    adjacency = _vertex_adjacency(faces_np, vertex_count)
    matrix = lil_matrix((len(free), len(free)), dtype=np.float64)
    rhs = np.zeros((len(free), values_np.shape[0] * 2), dtype=np.float64)
    flattened_values = values_np.transpose(1, 0, 2).reshape(len(constrained_np), -1)
    for row, vertex in enumerate(free):
        neighbors = adjacency[int(vertex)]
        if not neighbors:
            raise ValueError("a free vertex has no neighbors in the harmonic solve")
        matrix[row, row] = float(len(neighbors))
        for neighbor in neighbors:
            if is_constrained[neighbor]:
                rhs[row] += flattened_values[constrained_row[neighbor]]
            else:
                matrix[row, free_row[neighbor]] -= 1.0

    solved = spsolve(matrix.tocsr(), rhs)
    if solved.ndim == 1:
        solved = solved[:, None]
    modes = np.zeros((values_np.shape[0], vertex_count, 2), dtype=np.float64)
    modes[:, constrained_np] = values_np
    modes[:, free] = solved.reshape(len(free), values_np.shape[0], 2).transpose(1, 0, 2)
    return torch.as_tensor(
        modes,
        dtype=constrained_values.dtype,
        device=constrained_values.device,
    )


def _scaffold_mode_constraints(
    source_modes: Tensor,
    source_boundary: Tensor,
    outer_boundary: Tensor,
    transition_rings: Tensor | None,
    transition_weights: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Assemble optional geometry-aware multiring displacement constraints."""
    mode_count, boundary_count, dimension = source_modes.shape
    if dimension != 2:
        raise ValueError("source modes must contain 2D displacements")
    zero_outer = torch.zeros(
        (mode_count, outer_boundary.numel(), 2),
        dtype=source_modes.dtype,
        device=source_modes.device,
    )
    if transition_rings is None and transition_weights is None:
        return (
            torch.cat((source_boundary, outer_boundary)),
            torch.cat((source_modes, zero_outer), dim=1),
        )
    if transition_rings is None or transition_weights is None:
        raise ValueError("transition rings and weights must be supplied together")
    transition_rings = transition_rings.to(
        device=source_modes.device, dtype=torch.long
    )
    transition_weights = transition_weights.to(
        device=source_modes.device, dtype=source_modes.dtype
    )
    if transition_rings.ndim != 2 or transition_rings.shape[1] != boundary_count:
        raise ValueError("transition_rings must have shape [ring_count, boundary_count]")
    if transition_weights.shape != (transition_rings.shape[0],):
        raise ValueError("transition_weights must have one value per transition ring")
    if bool(
        torch.any(~torch.isfinite(transition_weights)).detach()
        or torch.any((transition_weights <= 0.0) | (transition_weights >= 1.0)).detach()
    ):
        raise ValueError("transition weights must be finite and lie strictly in (0, 1)")
    transition_values = (
        source_modes[:, None, :, :]
        * transition_weights[None, :, None, None]
    ).reshape(mode_count, -1, 2)
    return (
        torch.cat(
            (source_boundary, transition_rings.reshape(-1), outer_boundary)
        ),
        torch.cat((source_modes, transition_values, zero_outer), dim=1),
    )


def build_radial_fourier_harmonic_modes(
    initial_uv: Tensor,
    faces: Tensor,
    source_boundary: Tensor,
    outer_boundary: Tensor,
    frequencies: Iterable[int] = (2, 3),
    *,
    transition_rings: Tensor | None = None,
    transition_weights: Tensor | None = None,
) -> tuple[tuple[str, ...], Tensor]:
    """Build cosine/sine radial modes and extend them through the mesh."""
    frequencies = tuple(int(value) for value in frequencies)
    if not frequencies or any(value <= 0 for value in frequencies):
        raise ValueError("Fourier frequencies must be positive integers")
    source_boundary = source_boundary.to(device=initial_uv.device, dtype=torch.long)
    outer_boundary = outer_boundary.to(device=initial_uv.device, dtype=torch.long)
    source_uv = initial_uv[source_boundary]
    center = source_uv.mean(dim=0)
    radial = source_uv - center
    radii = torch.linalg.vector_norm(radial, dim=-1)
    if bool(torch.any(radii.detach() <= 0.0)):
        raise ValueError("source boundary contains the selected radial center")
    unit_radial = radial / radii[:, None]
    theta = torch.atan2(radial[:, 1], radial[:, 0])

    names: list[str] = []
    source_values: list[Tensor] = []
    for frequency in frequencies:
        names.extend((f"radial_cos_{frequency}", f"radial_sin_{frequency}"))
        source_values.extend(
            (
                torch.cos(frequency * theta)[:, None] * unit_radial,
                torch.sin(frequency * theta)[:, None] * unit_radial,
            )
        )
    source_modes = torch.stack(source_values)
    constrained, constrained_values = _scaffold_mode_constraints(
        source_modes,
        source_boundary,
        outer_boundary,
        transition_rings,
        transition_weights,
    )
    modes = harmonic_extend_boundary_values(
        faces,
        initial_uv.shape[0],
        constrained,
        constrained_values,
    )

    # One coefficient unit means one RMS UV unit on the source boundary.
    source_rms = torch.sqrt(
        torch.mean(torch.sum(modes[:, source_boundary].square(), dim=-1), dim=-1)
    )
    modes = modes / source_rms[:, None, None].clamp_min(1.0e-15)
    if not torch.allclose(
        modes[:, outer_boundary],
        torch.zeros_like(modes[:, outer_boundary]),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("harmonic modes unexpectedly move the fixed outer boundary")
    return tuple(names), modes


def _normalized_boundary_arclength(
    initial_uv: Tensor,
    source_boundary: Tensor,
) -> Tensor:
    source_uv = initial_uv[source_boundary]
    edge_lengths = torch.linalg.vector_norm(
        torch.roll(source_uv, shifts=-1, dims=0) - source_uv,
        dim=-1,
    )
    perimeter = edge_lengths.sum()
    if not bool(torch.isfinite(perimeter.detach())) or float(perimeter.detach()) <= 0.0:
        raise ValueError("source boundary has zero or non-finite perimeter")
    return torch.cat(
        (
            torch.zeros(1, dtype=initial_uv.dtype, device=initial_uv.device),
            torch.cumsum(edge_lengths[:-1], dim=0),
        )
    ) / perimeter


def _boundary_hat_weights(arclength: Tensor, control_count: int) -> Tensor:
    centers = torch.arange(
        control_count,
        dtype=arclength.dtype,
        device=arclength.device,
    ) / float(control_count)
    distance = torch.abs(arclength[:, None] - centers[None, :])
    circular_distance = torch.minimum(distance, 1.0 - distance)
    return (1.0 - float(control_count) * circular_distance).clamp_min(0.0)


def boundary_hat_support_diagnostics(
    initial_uv: Tensor,
    source_boundary: Tensor,
    control_counts: Iterable[int],
    *,
    minimum_peak_weight: float = 0.05,
) -> list[dict[str, object]]:
    """Measure whether every requested boundary hat is sampled meaningfully.

    A control scale is usable only when every scalar hat has at least one
    source-boundary sample and its largest sampled value is not vanishingly
    small.  This is stricter than merely requiring ``m <= boundary_count`` and
    catches non-uniform arclength sampling before the harmonic solve.
    """
    control_counts = tuple(int(value) for value in control_counts)
    if any(value < 3 for value in control_counts):
        raise ValueError("boundary hat control counts must be integers >= 3")
    if len(set(control_counts)) != len(control_counts):
        raise ValueError("boundary hat control counts must be unique")
    if not 0.0 <= minimum_peak_weight <= 1.0:
        raise ValueError("minimum_peak_weight must lie in [0, 1]")
    source_boundary = source_boundary.to(device=initial_uv.device, dtype=torch.long)
    arclength = _normalized_boundary_arclength(initial_uv, source_boundary)
    tolerance = 64.0 * torch.finfo(initial_uv.dtype).eps
    diagnostics: list[dict[str, object]] = []
    for control_count in control_counts:
        if control_count > source_boundary.numel():
            diagnostics.append(
                {
                    "control_count": control_count,
                    "supported": False,
                    "reason": "more_controls_than_boundary_vertices",
                    "minimum_peak_weight": 0.0,
                    "minimum_active_vertices": 0,
                    "unsupported_control_indices": list(range(control_count)),
                }
            )
            continue
        weights = _boundary_hat_weights(arclength, control_count)
        peak_weights = weights.amax(dim=0)
        active_vertices = torch.count_nonzero(weights > tolerance, dim=0)
        unsupported = torch.nonzero(
            peak_weights < minimum_peak_weight,
            as_tuple=False,
        ).flatten()
        diagnostics.append(
            {
                "control_count": control_count,
                "supported": unsupported.numel() == 0,
                "reason": (
                    "supported"
                    if unsupported.numel() == 0
                    else "insufficient_sampled_support"
                ),
                "minimum_peak_weight": float(peak_weights.amin().detach()),
                "minimum_active_vertices": int(active_vertices.amin().detach()),
                "unsupported_control_indices": [
                    int(value) for value in unsupported.detach().cpu().tolist()
                ],
            }
        )
    return diagnostics


def select_supported_boundary_hat_counts(
    initial_uv: Tensor,
    source_boundary: Tensor,
    control_counts: Iterable[int],
    *,
    minimum_peak_weight: float = 0.05,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    """Return requested scales whose every hat passes the support audit."""
    diagnostics = boundary_hat_support_diagnostics(
        initial_uv,
        source_boundary,
        control_counts,
        minimum_peak_weight=minimum_peak_weight,
    )
    selected = tuple(
        int(row["control_count"]) for row in diagnostics if bool(row["supported"])
    )
    return selected, diagnostics


def build_piecewise_linear_boundary_harmonic_modes(
    initial_uv: Tensor,
    faces: Tensor,
    source_boundary: Tensor,
    outer_boundary: Tensor,
    control_counts: Iterable[int] = (4, 8, 16),
    *,
    transition_rings: Tensor | None = None,
    transition_weights: Tensor | None = None,
) -> tuple[tuple[str, ...], Tensor]:
    """Build multiscale C0 boundary-hat modes with harmonic interiors.

    At a scale with ``m`` controls, every scalar hat equals one at its control
    location and decreases linearly to zero at the two neighboring controls
    in circular arclength.  Pairing every hat with the global x/y directions
    permits polygonal corners instead of imposing a smooth radial boundary.
    The mapped boundary edges remain linear, hence C0 continuous.
    """
    control_counts = tuple(int(value) for value in control_counts)
    if not control_counts or any(value < 3 for value in control_counts):
        raise ValueError("boundary hat control counts must be integers >= 3")
    if len(set(control_counts)) != len(control_counts):
        raise ValueError("boundary hat control counts must be unique")
    source_boundary = source_boundary.to(device=initial_uv.device, dtype=torch.long)
    outer_boundary = outer_boundary.to(device=initial_uv.device, dtype=torch.long)
    if any(value > source_boundary.numel() for value in control_counts):
        raise ValueError("a boundary hat scale has more controls than boundary vertices")

    arclength = _normalized_boundary_arclength(initial_uv, source_boundary)

    names: list[str] = []
    source_values: list[Tensor] = []
    directions = torch.eye(2, dtype=initial_uv.dtype, device=initial_uv.device)
    for control_count in control_counts:
        weights = _boundary_hat_weights(arclength, control_count)
        for control_index, weight in enumerate(weights.unbind(dim=1)):
            for axis, direction in zip(("x", "y"), directions, strict=True):
                names.append(f"boundary_hat_m{control_count}_c{control_index}_{axis}")
                source_values.append(weight[:, None] * direction)

    source_modes = torch.stack(source_values)
    constrained, constrained_values = _scaffold_mode_constraints(
        source_modes,
        source_boundary,
        outer_boundary,
        transition_rings,
        transition_weights,
    )
    modes = harmonic_extend_boundary_values(
        faces,
        initial_uv.shape[0],
        constrained,
        constrained_values,
    )
    source_rms = torch.sqrt(
        torch.mean(torch.sum(modes[:, source_boundary].square(), dim=-1), dim=-1)
    )
    if bool(torch.any(source_rms.detach() <= 0.0)):
        raise RuntimeError("a boundary hat mode vanished on the sampled boundary")
    modes = modes / source_rms[:, None, None]
    if not torch.equal(
        modes[:, outer_boundary], torch.zeros_like(modes[:, outer_boundary])
    ):
        raise RuntimeError("boundary hat modes unexpectedly move the fixed outer boundary")
    return tuple(names), modes


def mode_coordinate(uv: Tensor, mode: Tensor) -> tuple[Tensor, Tensor]:
    denominator = torch.sum(mode.square()).clamp_min(1.0e-30)
    coefficient = torch.sum(uv * mode) / denominator
    residual = uv - coefficient * mode
    return coefficient, residual


def _det(first: Tensor, second: Tensor) -> Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def mode_area_polynomials(
    residual: Tensor,
    mode: Tensor,
    faces: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``alpha, beta, gamma`` for every double-area polynomial."""
    triangle_residual = residual[faces]
    triangle_mode = mode[faces]
    r10 = triangle_residual[:, 1] - triangle_residual[:, 0]
    r20 = triangle_residual[:, 2] - triangle_residual[:, 0]
    d10 = triangle_mode[:, 1] - triangle_mode[:, 0]
    d20 = triangle_mode[:, 2] - triangle_mode[:, 0]
    alpha = _det(d10, d20)
    beta = _det(d10, r20) + _det(r10, d20)
    gamma = _det(r10, r20)
    return alpha, beta, gamma


def legal_mode_interval(
    uv: Tensor,
    mode: Tensor,
    faces: Tensor,
    area_floor: Tensor,
) -> ModeInterval:
    """Intersect exact scalar quadratic positivity components containing ``a``."""
    coefficient, residual = mode_coordinate(uv, mode)
    alpha, beta, gamma = mode_area_polynomials(residual, mode, faces)
    constant = gamma - area_floor
    current_margin = alpha * coefficient.square() + beta * coefficient + constant
    if bool(torch.any(current_margin.detach() <= 0.0)):
        raise ValueError("the current map is outside a harmonic mode's legal set")

    scale = torch.maximum(
        torch.maximum(torch.abs(alpha), torch.abs(beta)),
        torch.abs(constant),
    ).clamp_min(1.0)
    tolerance = 256.0 * torch.finfo(uv.dtype).eps * scale
    linear = torch.abs(alpha) <= tolerance
    beta_nonzero = torch.abs(beta) > tolerance
    linear_root = -constant / torch.where(beta_nonzero, beta, torch.ones_like(beta))
    negative_inf = torch.full_like(alpha, -torch.inf)
    positive_inf = torch.full_like(alpha, torch.inf)
    lower = torch.where(linear & beta_nonzero & (beta > 0.0), linear_root, negative_inf)
    upper = torch.where(linear & beta_nonzero & (beta < 0.0), linear_root, positive_inf)

    quadratic = ~linear
    discriminant = beta.square() - 4.0 * alpha * constant
    has_roots = quadratic & (discriminant > tolerance.square())
    # Inactive branches must remain finite because autograd can otherwise see
    # NaNs from 0/0 even though the following masks do not select those roots.
    safe_discriminant = torch.where(
        has_roots, discriminant, torch.ones_like(discriminant)
    )
    safe_alpha = torch.where(quadratic, alpha, torch.ones_like(alpha))
    sqrt_discriminant = torch.sqrt(safe_discriminant)
    first = (-beta - sqrt_discriminant) / (2.0 * safe_alpha)
    second = (-beta + sqrt_discriminant) / (2.0 * safe_alpha)
    root_low = torch.minimum(first, second)
    root_high = torch.maximum(first, second)

    concave = has_roots & (alpha < 0.0)
    lower = torch.where(concave, root_low, lower)
    upper = torch.where(concave, root_high, upper)

    convex = has_roots & (alpha > 0.0)
    convex_left = convex & (coefficient < root_low)
    convex_right = convex & (coefficient > root_high)
    upper = torch.where(convex_left, root_low, upper)
    lower = torch.where(convex_right, root_high, lower)

    # A concave quadratic without real roots cannot be positive.  The current
    # margin check above should make this reachable only through roundoff.
    invalid_no_roots = quadratic & ~has_roots & (alpha < 0.0)
    if bool(torch.any(invalid_no_roots.detach())):
        raise ValueError("encountered an inconsistent concave area polynomial")

    global_lower = lower.amax()
    global_upper = upper.amin()
    if not bool(
        (global_lower.detach() < coefficient.detach())
        and (coefficient.detach() < global_upper.detach())
    ):
        raise ValueError("empty harmonic mode interval around the current coefficient")
    return ModeInterval(global_lower, global_upper, coefficient)


def interval_to_real(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    lower_finite = bool(torch.isfinite(lower.detach()))
    upper_finite = bool(torch.isfinite(upper.detach()))
    if lower_finite and upper_finite:
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        normalized = (value - midpoint) / half_width
        if bool(torch.abs(normalized.detach()) >= 1.0):
            raise ValueError("coefficient lies outside its finite legal interval")
        return torch.atanh(normalized)
    if lower_finite:
        return torch.log(value - lower)
    if upper_finite:
        return -torch.log(upper - value)
    return value


def real_to_interval(latent: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    lower_finite = bool(torch.isfinite(lower.detach()))
    upper_finite = bool(torch.isfinite(upper.detach()))
    if lower_finite and upper_finite:
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        return midpoint + half_width * torch.tanh(latent)
    if lower_finite:
        return lower + torch.exp(latent)
    if upper_finite:
        return upper - torch.exp(-latent)
    return latent


def interval_q(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """A dimensionless 0-at-center, 1-at-boundary conditioning diagnostic."""
    if bool(torch.isfinite(lower.detach()) and torch.isfinite(upper.detach())):
        return torch.abs((2.0 * value - lower - upper) / (upper - lower))
    return torch.zeros((), dtype=value.dtype, device=value.device)


class GlobalHarmonicModeFlow(nn.Module):
    """A sequence of exact scalar coupling layers along harmonic mesh modes."""

    def __init__(
        self,
        initial_uv: Tensor,
        faces: Tensor,
        modes: Tensor,
        names: Iterable[str],
        *,
        cycles: int = 2,
        max_log_scale: float = 0.08,
        max_shift: float = 0.20,
        area_margin_ratio: float = 1.0e-6,
    ) -> None:
        super().__init__()
        names = tuple(names)
        if modes.shape != (len(names), initial_uv.shape[0], 2):
            raise ValueError("modes and names do not match the initial mesh")
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if max_log_scale < 0.0 or max_shift < 0.0:
            raise ValueError("coupling bounds must be non-negative")
        if not 0.0 <= area_margin_ratio < 1.0:
            raise ValueError("area_margin_ratio must lie in [0, 1)")
        initial_areas = signed_double_areas(initial_uv, faces)
        if bool(torch.any(initial_areas.detach() <= 0.0)):
            raise ValueError("global harmonic modes require a positive initial embedding")

        self.register_buffer("initial_uv", initial_uv.clone())
        self.register_buffer("faces", faces.clone())
        self.register_buffer("modes", modes.clone())
        self.register_buffer("area_floor", area_margin_ratio * initial_areas)
        self.base_names = names
        self.cycles = int(cycles)
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)
        self.parameters_raw = nn.Parameter(
            torch.zeros(self.cycles, len(names), 2, dtype=initial_uv.dtype)
        )

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(
            f"cycle_{cycle}:{name}"
            for cycle in range(self.cycles)
            for name in self.base_names
        )

    def _layer(
        self,
        uv: Tensor,
        mode_index: int,
        raw: Tensor,
        *,
        inverse: bool,
        interval: ModeInterval | None = None,
    ) -> tuple[Tensor, ModeInterval, Tensor, Tensor, Tensor]:
        mode = self.modes[mode_index]
        if interval is None:
            interval = legal_mode_interval(uv, mode, self.faces, self.area_floor)
        coefficient, residual = mode_coordinate(uv, mode)
        latent = interval_to_real(coefficient, interval.lower, interval.upper)
        log_scale = self.max_log_scale * torch.tanh(raw[0])
        shift = self.max_shift * torch.tanh(raw[1])
        transformed = (
            (latent - shift) * torch.exp(-log_scale)
            if inverse
            else latent * torch.exp(log_scale) + shift
        )
        mapped_coefficient = real_to_interval(
            transformed, interval.lower, interval.upper
        )
        mapped = residual + mapped_coefficient * mode
        q = interval_q(mapped_coefficient, interval.lower, interval.upper)
        minimum_area = signed_double_areas(mapped, self.faces).amin()
        return mapped, interval, mapped_coefficient, q, minimum_area

    def forward(
        self,
        uv: Tensor | None = None,
        *,
        inverse: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, HarmonicModeDiagnostics]:
        result = self.initial_uv if uv is None else uv
        before_values: list[Tensor] = []
        after_values: list[Tensor] = []
        lower_values: list[Tensor] = []
        upper_values: list[Tensor] = []
        q_values: list[Tensor] = []
        minimum_areas: list[Tensor] = []
        names: list[str] = []
        cycle_indices = (
            range(self.cycles - 1, -1, -1) if inverse else range(self.cycles)
        )
        for cycle in cycle_indices:
            mode_indices = (
                range(len(self.base_names) - 1, -1, -1)
                if inverse
                else range(len(self.base_names))
            )
            for mode_index in mode_indices:
                interval = legal_mode_interval(
                    result,
                    self.modes[mode_index],
                    self.faces,
                    self.area_floor,
                )
                before = interval.coefficient
                result, interval, after, q, minimum_area = self._layer(
                    result,
                    mode_index,
                    self.parameters_raw[cycle, mode_index],
                    inverse=inverse,
                    interval=interval,
                )
                if return_diagnostics:
                    names.append(f"cycle_{cycle}:{self.base_names[mode_index]}")
                    before_values.append(before)
                    after_values.append(after)
                    lower_values.append(interval.lower)
                    upper_values.append(interval.upper)
                    q_values.append(q)
                    minimum_areas.append(minimum_area)
        if not return_diagnostics:
            return result
        return result, HarmonicModeDiagnostics(
            tuple(names),
            torch.stack(before_values),
            torch.stack(after_values),
            torch.stack(lower_values),
            torch.stack(upper_values),
            torch.stack(q_values),
            torch.stack(minimum_areas),
        )


def preflight_mode_intervals(
    uv: Tensor,
    faces: Tensor,
    modes: Tensor,
    names: Iterable[str],
    *,
    area_margin_ratio: float = 1.0e-6,
) -> list[dict[str, float | str | bool]]:
    initial_areas = signed_double_areas(uv, faces)
    floor = area_margin_ratio * initial_areas
    rows: list[dict[str, float | str | bool]] = []
    for name, mode in zip(names, modes):
        interval = legal_mode_interval(uv, mode, faces, floor)
        lower_finite = bool(torch.isfinite(interval.lower.detach()))
        upper_finite = bool(torch.isfinite(interval.upper.detach()))
        left = (
            float((interval.coefficient - interval.lower).detach())
            if lower_finite
            else math.inf
        )
        right = (
            float((interval.upper - interval.coefficient).detach())
            if upper_finite
            else math.inf
        )
        rows.append(
            {
                "name": str(name),
                "coefficient": float(interval.coefficient.detach()),
                "lower": float(interval.lower.detach()),
                "upper": float(interval.upper.detach()),
                "left_clearance": left,
                "right_clearance": right,
                "minimum_clearance": min(left, right),
                "bounded": lower_finite and upper_finite,
            }
        )
    return rows
