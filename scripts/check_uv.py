from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_nvp.injectivity.validators import validate_uv
from surface_nvp.io import load_mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prim-path", default=None)
    args = parser.parse_args()
    mesh = load_mesh(args.input, prim_path=args.prim_path)
    if mesh.uv is None:
        raise ValueError("mesh has no UV")
    print(validate_uv(mesh.uv, mesh.faces))


if __name__ == "__main__":
    main()
