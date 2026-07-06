from __future__ import annotations

from pathlib import Path

from .mesh_data import MeshData
from .obj_io import load_obj, save_obj


def load_mesh(path: str | Path, prim_path: str | None = None) -> MeshData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix in {".usd", ".usda", ".usdc"}:
        from .usd_io import load_usd
        return load_usd(path, prim_path=prim_path)
    raise ValueError(f"unsupported mesh format: {suffix}")


def save_mesh(path: str | Path, mesh: MeshData, uv=None, prim_path: str | None = None) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        save_obj(path, mesh, uv=uv)
        return
    if suffix in {".usd", ".usda", ".usdc"}:
        from .usd_io import save_usd
        save_usd(path, mesh, uv=uv, prim_path=prim_path or mesh.prim_path or "/Mesh")
        return
    raise ValueError(f"unsupported mesh format: {suffix}")
