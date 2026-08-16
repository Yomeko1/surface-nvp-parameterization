# SLIM Runner

This optional executable wraps libigl's SLIM implementation so that it starts
from the exact UV coordinates stored in the input OBJ. It is kept separate from
the Python package and fetches libigl v2.6.0 while configuring the build.

## Build

```bash
cmake -S external/slim_runner -B build/slim_runner
cmake --build build/slim_runner --config Release
```

On Windows, the executable is normally written to:

```text
build/slim_runner/Release/surface_nvp_slim.exe
```

On single-config generators, it is normally written to:

```text
build/slim_runner/surface_nvp_slim
```

The wrapper and the SLIM source files used from libigl are under the Mozilla
Public License 2.0. See libigl's source headers and license files for details.

## Protocol

```text
surface_nvp_slim INPUT.obj OUTPUT.obj ITERATIONS
```

`INPUT.obj` must be triangular and use matching one-to-one vertex and UV
indices. The Python `scripts/run_slim.py` command creates this input format
automatically.
