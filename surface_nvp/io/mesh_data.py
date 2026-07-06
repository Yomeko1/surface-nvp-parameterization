from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    uv: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    prim_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.uv is not None:
            self.uv = np.asarray(self.uv, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [V, 3]")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape [F, 3]; triangulate the mesh first")
        if self.uv is not None and (self.uv.ndim != 2 or self.uv.shape[1] != 2):
            raise ValueError("uv must have shape [V, 2]")

    @property
    def has_uv(self) -> bool:
        return self.uv is not None
