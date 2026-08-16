from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.init_param import normalize_uv_geometry_scale, tutte_parameterize
from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.visualization.plot_uv import save_uv_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="tutte", choices=["tutte"])
    parser.add_argument("--boundary", default="circle", choices=["circle", "square"])
    parser.add_argument("--geometry-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()

    mesh = load_mesh(args.input, prim_path=args.prim_path)
    uv = tutte_parameterize(mesh.vertices, mesh.faces, boundary_mode=args.boundary)
    if args.geometry_scale:
        uv = normalize_uv_geometry_scale(mesh.vertices, mesh.faces, uv)
    metrics = validate_uv(uv, mesh.faces)
    print(metrics)
    save_mesh(args.output, mesh, uv=uv)
    save_uv_plot(Path(args.output).with_suffix(".uv.png"), uv, mesh.faces)


if __name__ == "__main__":
    main()
