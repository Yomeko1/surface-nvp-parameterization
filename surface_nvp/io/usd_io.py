from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh_data import MeshData


def _require_pxr():
    try:
        from pxr import Sdf, Usd, UsdGeom  # type: ignore
    except ImportError as exc:
        raise ImportError("USD support requires the optional package 'usd-core' with pxr bindings") from exc
    return Sdf, Usd, UsdGeom


def _find_first_mesh(stage, UsdGeom):
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return prim
    return None


def load_usd(path: str | Path, prim_path: str | None = None) -> MeshData:
    _, Usd, UsdGeom = _require_pxr()
    path = Path(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"failed to open USD file: {path}")
    prim = stage.GetPrimAtPath(prim_path) if prim_path else _find_first_mesh(stage, UsdGeom)
    if prim is None or not prim.IsValid():
        raise ValueError("no UsdGeom.Mesh found")
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    faces = []
    cursor = 0
    for count in counts:
        face = indices[cursor:cursor + count]
        cursor += count
        if count < 3:
            continue
        for k in range(1, count - 1):
            faces.append([face[0], face[k], face[k + 1]])
    uv = None
    primvars = UsdGeom.PrimvarsAPI(prim)
    st = primvars.GetPrimvar("st")
    if st and st.HasValue():
        st_values = np.asarray(st.Get(), dtype=np.float64)
        indices_attr = st.GetIndicesAttr()
        if indices_attr and indices_attr.HasValue():
            st_indices = np.asarray(indices_attr.Get(), dtype=np.int64)
            uv_accum = np.zeros((vertices.shape[0], 2), dtype=np.float64)
            uv_count = np.zeros((vertices.shape[0], 1), dtype=np.float64)
            for vi, ti in zip(indices, st_indices):
                uv_accum[vi] += st_values[ti]
                uv_count[vi] += 1.0
            valid = uv_count[:, 0] > 0
            uv = np.zeros((vertices.shape[0], 2), dtype=np.float64)
            uv[valid] = uv_accum[valid] / uv_count[valid]
        elif st_values.shape[0] == vertices.shape[0]:
            uv = st_values[:, :2]
    return MeshData(vertices, np.asarray(faces), uv=uv, source_path=str(path), prim_path=str(prim.GetPath()))


def save_usd(path: str | Path, mesh: MeshData, uv: np.ndarray | None = None, prim_path: str = "/Mesh") -> None:
    Sdf, Usd, UsdGeom = _require_pxr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uv = mesh.uv if uv is None else uv
    stage = Usd.Stage.CreateNew(str(path))
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    usd_mesh.GetPointsAttr().Set(mesh.vertices.tolist())
    usd_mesh.GetFaceVertexCountsAttr().Set([3] * int(mesh.faces.shape[0]))
    usd_mesh.GetFaceVertexIndicesAttr().Set(mesh.faces.reshape(-1).astype(int).tolist())
    if uv is not None:
        primvar = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(uv.astype(np.float32).tolist())
    stage.GetRootLayer().Save()
