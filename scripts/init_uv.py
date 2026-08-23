from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.init_param import generate_initial_uv
from surface_nvp.io import load_mesh, save_mesh
from surface_nvp.visualization.plot_uv import save_uv_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="tutte", choices=["tutte", "mean_value", "abfpp", "auto"])
    parser.add_argument("--boundary", default="circle", choices=["circle", "square"])
    parser.add_argument("--abfpp-executable", default=None)
    parser.add_argument("--geometry-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()

    mesh = load_mesh(args.input, prim_path=args.prim_path)
    result = generate_initial_uv(
        mesh,
        method=args.method,
        boundary_mode=args.boundary,
        geometry_scale=args.geometry_scale,
        abfpp_executable=args.abfpp_executable,
    )
    payload = {
        "requested_method": result.requested_method,
        "selected_method": result.selected_method,
        "boundary": args.boundary,
        "geometry_scale": args.geometry_scale,
        "candidates": result.candidates,
    }
    print(json.dumps(payload, indent=2))
    output = Path(args.output)
    save_mesh(output, mesh, uv=result.uv)
    save_uv_plot(output.with_suffix(".uv.png"), result.uv, mesh.faces)
    output.with_suffix(".init.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
