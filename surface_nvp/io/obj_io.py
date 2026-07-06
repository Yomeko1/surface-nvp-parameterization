from __future__ import annotations

from pathlib import Path
import shutil
from typing import List, Tuple

import numpy as np

from .mesh_data import MeshData


def _parse_face_vertex(token: str) -> Tuple[int, int | None]:
    parts = token.split("/")
    vi = int(parts[0]) - 1
    ti = None
    if len(parts) > 1 and parts[1] != "":
        ti = int(parts[1]) - 1
    return vi, ti


def load_obj(path: str | Path) -> MeshData:
    path = Path(path)
    vertices: List[List[float]] = []
    texcoords: List[List[float]] = []
    faces: List[List[int]] = []
    face_tex: List[List[int | None]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "v":
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "vt":
                texcoords.append([float(fields[1]), float(fields[2])])
            elif fields[0] == "f":
                parsed = [_parse_face_vertex(tok) for tok in fields[1:]]
                if len(parsed) < 3:
                    continue
                # Fan triangulate polygons.
                for k in range(1, len(parsed) - 1):
                    tri = [parsed[0], parsed[k], parsed[k + 1]]
                    faces.append([p[0] for p in tri])
                    face_tex.append([p[1] for p in tri])

    uv = None
    if texcoords and face_tex:
        uv_accum = np.zeros((len(vertices), 2), dtype=np.float64)
        uv_count = np.zeros((len(vertices), 1), dtype=np.float64)
        for tri, tri_tex in zip(faces, face_tex):
            for vi, ti in zip(tri, tri_tex):
                if ti is not None:
                    uv_accum[vi] += np.asarray(texcoords[ti], dtype=np.float64)
                    uv_count[vi] += 1.0
        valid = uv_count[:, 0] > 0
        if np.any(valid):
            uv = np.zeros((len(vertices), 2), dtype=np.float64)
            uv[valid] = uv_accum[valid] / uv_count[valid]

    return MeshData(np.asarray(vertices), np.asarray(faces), uv=uv, source_path=str(path))


def save_obj(path: str | Path, mesh: MeshData, uv: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uv = mesh.uv if uv is None else uv
    display = _read_obj_display_data(mesh.source_path, len(mesh.vertices))
    _copy_material_files(display["mtllib"], mesh.source_path, path)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Written by surface_nvp_param\n")
        for mtllib in display["mtllib"]:
            f.write(f"mtllib {mtllib}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        if uv is not None:
            for u in uv:
                f.write(f"vt {u[0]:.17g} {u[1]:.17g}\n")
        for vn in display["normals"]:
            f.write(f"vn {vn[0]:.17g} {vn[1]:.17g} {vn[2]:.17g}\n")
        for usemtl in display["usemtl"]:
            f.write(f"usemtl {usemtl}\n")
        for face in mesh.faces:
            if uv is not None:
                a, b, c = face + 1
                if display["vertex_normals"]:
                    f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                else:
                    f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
            else:
                a, b, c = face + 1
                if display["vertex_normals"]:
                    f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
                else:
                    f.write(f"f {a} {b} {c}\n")


def _read_obj_display_data(source_path: str | None, num_vertices: int) -> dict:
    data = {"mtllib": [], "usemtl": [], "normals": [], "vertex_normals": False}
    if source_path is None:
        return data
    source = Path(source_path)
    if source.suffix.lower() != ".obj" or not source.exists():
        return data

    first_face_tokens = None
    with source.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if fields[0] == "mtllib":
                data["mtllib"].extend(fields[1:])
            elif fields[0] == "usemtl" and len(fields) > 1:
                if fields[1] not in data["usemtl"]:
                    data["usemtl"].append(fields[1])
            elif fields[0] == "vn":
                data["normals"].append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and first_face_tokens is None:
                first_face_tokens = fields[1:]

    normals = np.asarray(data["normals"], dtype=np.float64)
    data["normals"] = normals if len(normals) == num_vertices else np.empty((0, 3), dtype=np.float64)
    data["vertex_normals"] = len(data["normals"]) == num_vertices and _face_uses_matching_vertex_normals(first_face_tokens)
    return data


def _face_uses_matching_vertex_normals(tokens: list[str] | None) -> bool:
    if not tokens:
        return False
    for token in tokens:
        parts = token.split("/")
        if len(parts) < 3 or parts[2] == "":
            return False
        if int(parts[0]) != int(parts[2]):
            return False
    return True


def _copy_material_files(mtllibs: list[str], source_path: str | None, output_path: Path) -> None:
    if source_path is None:
        return
    source_dir = Path(source_path).parent
    for mtllib in mtllibs:
        source_mtl = source_dir / mtllib
        dest_mtl = output_path.parent / Path(mtllib).name
        if source_mtl.exists() and source_mtl.resolve() != dest_mtl.resolve():
            shutil.copyfile(source_mtl, dest_mtl)
