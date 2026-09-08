# Mesh-aligned PL-NVP v3.1

This directory contains the isolated research implementation of mesh-aligned
piecewise-linear NVP. It does not modify the retained v2.4 Affine/Spline
pipeline. The current trainer has no rollback branch: legality is enforced by
the reachable set of every PL layer, while optimization minimizes only
original-face area-weighted symmetric Dirichlet distortion.

For the full derivation and paper links, read
[`PRINCIPLE_CN.md`](PRINCIPLE_CN.md). For reproducible Windows/Linux commands,
read [`RUN_PIPELINE_CN.md`](RUN_PIPELINE_CN.md).

## v3.1 architecture

```text
Tutte circle
  -> six-ring scaffold with fixed convex outer boundary
  -> H1 certified global harmonic/boundary modes (100 iterations)
  -> mesh-local legal-interval spline PL-NVP (500 iterations)
  -> H2 certified global harmonic refinement (100 iterations)
  -> original-mesh and extended-scaffold audit
```

The local layer updates proper-color independent vertex sets. For each active
vertex, the current one-ring defines an exact convex legal region. The v3.1
default applies a conditional rational-quadratic spline inside an exact legal
axis interval. H1/H2 coefficients likewise stay inside exact connected
positive-area intervals. A fixed simple scaffold boundary plus positive
extended-face orientation gives global injectivity for a disk map.

## Release configuration

The executable release preset is [`v3_1_default.yaml`](v3_1_default.yaml). It
contains the same defaults as `run_harmonic_local.py`; an explicit CLI option
overrides the YAML value.

Important defaults:

- Tutte circle with geometry scaling;
- H100–Spline500–H100;
- harmonic cycles 2, local cycles 4, hidden dimension 32;
- frequencies 2–5 and automatically supported C0 hats 4/8/16/32;
- eight spline bins, `atanh` conditioning model;
- six scaffold rings, scale 1.1, exponent 3, geometric profile;
- risk-adaptive LR reduction and conservative recovery;
- CUDA, float64, fixed seed 20260906;
- no rollback, barrier, intersection loss, scaffold quality loss, or tail loss.

## Run v3.1

From the repository root:

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/Balls/Balls.obj `
  --output-dir data/output/mesh_pl_nvp/v3.1/Balls
```

Use `--device cpu` if CUDA is unavailable. Use `--local-iters 50` and
`--final-harmonic-iters 0` for a smoke test. Existing output directories should
not be reused because result files are written with deterministic names.

Each run writes separate `two_stage/` and `three_stage/` directories containing
the mesh, model, actual config, full metrics/history, JSON/CSV summary, UV and
boundary comparisons, loss plot, distortion heatmap, and flip/intersection
diagnostics. `data/output/` is intentionally Git-ignored.

## Tests

```powershell
python -m pytest research/mesh_pl_nvp/tests -q
python -m pytest -q
```

The tests cover polytope maps, analytic centers, batched coupling, exact spline
intervals, harmonic mode intervals, multiring scaffolds, explicit inverse
composition, LR recovery, YAML configuration, topology checks, and pipeline
artifacts.

## Retained v3.0 entry points

`run_pipeline.py`, `default.yaml`, `run_balls_patch.py`, and `validate.py` are
retained for the v3.0 baseline and low-level diagnostics. The v3.1 release
entry point is `run_harmonic_local.py`; do not confuse its 100/500/100
three-stage budget with the older single-stage 1000-iteration affine preset.

## Current interpretation

The v3.1 configuration completed Balls, Cow, David328, Isis, and
NefertitiFace with zero original/scaffold flips and self-intersections and with
rollback disabled. Increasing the local stage from 200 to 500 reduced mean
area-weighted SD on all five meshes. LR recovery activated only on Cow and was
neutral on the other four models. Numerical accuracy and runtime are still not
uniformly superior to mature methods; v3.1 is a reproducible research release,
not a claim of state-of-the-art distortion.
