from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.injectivity.signed_area import triangle_signed_areas
from surface_nvp.io.mesh_data import MeshData
from surface_nvp.training.metrics import compute_distortion_metrics

from .abfpp import abfpp_parameterize
from .mean_value import mean_value_parameterize
from .tutte import tutte_parameterize


@dataclass
class InitialUVResult:
    uv: np.ndarray
    requested_method: str
    selected_method: str
    candidates: list[dict]


def generate_initial_uv(
    mesh: MeshData,
    method: str = "tutte",
    boundary_mode: str = "circle",
    geometry_scale: bool = False,
    abfpp_executable: str | Path | None = None,
) -> InitialUVResult:
    if method not in {"tutte", "mean_value", "abfpp", "auto"}:
        raise ValueError(f"unsupported init method: {method}")

    methods = ("tutte", "mean_value", "abfpp") if method == "auto" else (method,)
    candidates: list[dict] = []
    valid_results: list[tuple[float, str, np.ndarray]] = []
    for candidate_method in methods:
        try:
            uv = _generate_candidate(
                mesh,
                candidate_method,
                boundary_mode=boundary_mode,
                abfpp_executable=abfpp_executable,
            )
            if float(triangle_signed_areas(uv, mesh.faces).sum()) < 0.0:
                uv = uv.copy()
                uv[:, 1] *= -1.0
            if geometry_scale:
                from .initial_uv import normalize_uv_geometry_scale

                uv = normalize_uv_geometry_scale(mesh.vertices, mesh.faces, uv)
            injectivity = validate_uv(uv, mesh.faces)
            record = {"method": candidate_method, "injectivity": injectivity}
            if injectivity["is_valid"]:
                distortion = compute_distortion_metrics(mesh.vertices, mesh.faces, uv)
                score = float(distortion["symmetric_dirichlet_area_weighted_mean"])
                record["distortion"] = distortion
                record["selection_score"] = score
                valid_results.append((score, candidate_method, uv))
            candidates.append(record)
        except Exception as exc:
            candidates.append(
                {
                    "method": candidate_method,
                    "error": f"{type(exc).__name__}: {exc}",
                    "is_available": False,
                }
            )
            if method != "auto":
                raise

    if not valid_results:
        details = "; ".join(
            f"{entry['method']}: {entry.get('error', entry.get('injectivity'))}"
            for entry in candidates
        )
        raise ValueError(f"no valid initial UV candidate: {details}")
    _, selected_method, selected_uv = min(valid_results, key=lambda item: item[0])
    for candidate in candidates:
        candidate["selected"] = candidate["method"] == selected_method
    return InitialUVResult(
        uv=selected_uv,
        requested_method=method,
        selected_method=selected_method,
        candidates=candidates,
    )


def _generate_candidate(
    mesh: MeshData,
    method: str,
    boundary_mode: str,
    abfpp_executable: str | Path | None,
) -> np.ndarray:
    if method == "tutte":
        return tutte_parameterize(mesh.vertices, mesh.faces, boundary_mode=boundary_mode)
    if method == "mean_value":
        return mean_value_parameterize(mesh.vertices, mesh.faces, boundary_mode=boundary_mode)
    if method == "abfpp":
        return abfpp_parameterize(mesh.vertices, mesh.faces, executable=abfpp_executable)
    raise ValueError(f"unsupported init method: {method}")
