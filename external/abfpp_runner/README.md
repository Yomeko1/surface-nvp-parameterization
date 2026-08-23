# ABF++ Runner

This optional C++20 executable wraps OpenABF v2.1.0. It computes ABF++ angles and
reconstructs one UV coordinate per input vertex with angle-based LSCM.

## Build

```bash
cmake -S external/abfpp_runner -B build/abfpp_runner
cmake --build build/abfpp_runner --config Release
```

On Windows, the executable is normally written to:

```text
build/abfpp_runner/Release/surface_nvp_abfpp.exe
```

## Protocol

```text
surface_nvp_abfpp INPUT.obj OUTPUT.obj
```

The input must be a triangular, manifold disk. The output preserves vertex and
face order and stores one-to-one UV indices. The Python layer performs the v2.4
local-flip and global-intersection checks before accepting the result.

OpenABF is licensed under Apache License 2.0. Its `LICENSE` and `NOTICE` files
are fetched with the pinned v2.1.0 source during configuration.
