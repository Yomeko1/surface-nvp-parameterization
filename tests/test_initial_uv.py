import numpy as np
import pytest
import torch

from surface_nvp.init_param import normalize_uv_geometry_scale, resolve_initial_uv
from surface_nvp.io import save_mesh
from surface_nvp.io.mesh_data import MeshData
from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.losses.distortion import jacobian_determinants


def _mesh() -> MeshData:
    return MeshData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )


def test_resolve_initial_uv_from_explicit_mesh(tmp_path):
    mesh = _mesh()
    uv = np.array([[0.1, 0.2], [1.1, 0.2], [0.1, 1.2]])
    initial_path = tmp_path / "initial.obj"
    save_mesh(initial_path, mesh, uv=uv)

    actual = resolve_initial_uv(mesh, initial_uv_path=initial_path)

    np.testing.assert_allclose(actual, uv)


def test_resolve_initial_uv_rejects_different_topology(tmp_path):
    mesh = _mesh()
    initial = MeshData(mesh.vertices, np.array([[0, 2, 1]]))
    initial_path = tmp_path / "initial.obj"
    save_mesh(initial_path, initial, uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="same triangle topology"):
        resolve_initial_uv(mesh, initial_uv_path=initial_path)


def test_geometry_scale_normalizes_median_jacobian_and_preserves_validity():
    mesh = _mesh()
    uv = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

    scaled = normalize_uv_geometry_scale(mesh.vertices, mesh.faces, uv)
    determinants = jacobian_determinants(
        torch.as_tensor(mesh.vertices, dtype=torch.float64),
        torch.as_tensor(mesh.faces),
        torch.as_tensor(scaled, dtype=torch.float64),
    )

    assert float(determinants.abs().median()) == pytest.approx(1.0)
    assert validate_uv(scaled, mesh.faces)["is_valid"]
