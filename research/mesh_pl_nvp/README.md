# Mesh-aligned PL-NVP v3.0

This directory is the isolated v3.0 research implementation. Its model and trainer remain
outside the production `surface_nvp` package and do not modify the v2.4
affine/spline pipeline. The generic research runner deliberately reuses public
v2.4 utilities for mesh I/O, Tutte initialization, metrics, plots, and summaries
so that experiments have the same operating style and comparison fields.

For a step-by-step Chinese explanation of the mathematical construction and
its relation to NVP, see [`PRINCIPLE_CN.md`](PRINCIPLE_CN.md).

The Balls experiment report is kept with its ignored local outputs at
`data/output/mesh_pl_nvp/BALLS_EXPERIMENT_CN.md`.

For the generic OBJ/USD/USDA command and complete Chinese instructions, see
[`RUN_PIPELINE_CN.md`](RUN_PIPELINE_CN.md).

The unified 1000-step report for Balls, David328, Isis, and NefertitiFace is
kept with its ignored local outputs at
`data/output/mesh_pl_nvp/SIMPLE_MODELS_1000_CN.md`.

## What is implemented

- A radial homeomorphism from `R^2` to the interior of a bounded convex polygon
  `K = {x | A x < b}` and its analytical inverse `psi_K: K -> R^2`.
- A differentiable log-barrier analytic center for each changing polygon.
- Construction of the legal one-ring kernel of an interior mesh vertex.
- Proper vertex coloring and independent-set coupling updates.
- Forward and reverse color cycles with fixed boundary vertices.
- A same-color batched implementation for dynamic kernels, analytic centers,
  radial maps, and graph-conditioned coupling.
- Checks for signed triangle areas, proper non-adjacent edge intersections,
  round-trip error, automatic differentiation, and near-boundary conditioning.

The deterministic conditioner in `mesh_coupling.py` is only a test fixture. It
stands in for a future graph neural network and is intentionally strong enough
to expose near-boundary numerical conditioning.

## Run the tests

From the repository root:

```powershell
python -m pytest research/mesh_pl_nvp/tests -q
```

## Run the full validation

```powershell
python -m research.mesh_pl_nvp.validate
```

## Run the trainable Balls experiment

Local 8-ring disk patch:

```powershell
python -m research.mesh_pl_nvp.run_balls_patch --rings 8 --cycles 2 --iterations 300
```

Complete Balls disk:

```powershell
python -m research.mesh_pl_nvp.run_balls_patch --full-mesh --cycles 1 --iterations 50
```

Deeper batched experiment:

```powershell
python -m research.mesh_pl_nvp.run_balls_patch --full-mesh --cycles 4 --iterations 300
```

Long training followed by low-learning-rate refinement:

```powershell
python -m research.mesh_pl_nvp.run_balls_patch --full-mesh --cycles 4 --iterations 1000 --output research/mesh_pl_nvp/artifacts/balls_full_batched_4c_1000
python -m research.mesh_pl_nvp.run_balls_patch --full-mesh --cycles 4 --iterations 300 --learning-rate 0.0003 --initial-model-state research/mesh_pl_nvp/artifacts/balls_full_batched_4c_1000/model_state.pt --output research/mesh_pl_nvp/artifacts/balls_full_batched_4c_1000_refine_300
```

`metrics.json` includes the top distinct radial-fraction hotspots and their
one-rings; `q_hotspots.png` visualizes their locations. Use
`--conditioner-features local-geometry` for the optional static 3D feature
ablation. `basic` remains the default because it achieved the lower
area-weighted SD in the current 300-step comparison.

Mesh-adaptive plateau LR and free-boundary scaffold:

```powershell
python -m research.mesh_pl_nvp.run_balls_patch --full-mesh --cycles 4 --iterations 1000 --lr-schedule adaptive-plateau
python -m research.mesh_pl_nvp.run_balls_patch --full-mesh --scaffold --scaffold-scale 1.1 --cycles 4 --iterations 300
```

`adaptive-plateau` uses relative sliding-window improvement rather than a
mesh-specific fixed decay step. The scaffold adds a fixed convex outer ring,
turning the source boundary into movable interior vertices while preserving
the existing positive-face/global-injectivity argument on the extended disk.

The command writes the following ignored local artifacts:

- `research/mesh_pl_nvp/artifacts/validation_summary.json`
- `research/mesh_pl_nvp/artifacts/radial_mapping.png`
- `research/mesh_pl_nvp/artifacts/mesh_coupling.png`

To choose another output directory:

```powershell
python -m research.mesh_pl_nvp.validate --output F:\path\to\validation_output
```

## Run any disk mesh with v2.4-style I/O

The current research default is 4 cycles, hidden dimension 32, 1000 iterations,
adaptive plateau LR, and a scale-1.1 scaffold. If no `--initial-uv` is supplied,
the runner always recomputes a Tutte map even when the input file already has UV.
Mean-Value and ABF++ are intentionally not exposed here.

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/David328/David328.usda `
  --output data/output/mesh_pl_nvp/David328/David328_mesh_pl_nvp.usda
```

The output prefix receives the optimized mesh, config, metrics, JSON/CSV
summary, model state, UV comparison, distortion/area/intersection diagnostics,
loss curve, and q-hotspot plot. The training metadata explicitly records
`rollback_enabled: false`; any hard-validity failure stops the run.

## Interpretation

Membership in the open kernel gives a structural no-flip guarantee for every
updated one-ring. A proper coloring makes same-color updates independent. With
a fixed simple boundary, positive orientation of all faces gives a globally
injective piecewise-linear disk map.

The radial map approaches the polygon boundary as the latent radius tends to
infinity. Consequently, its inverse becomes ill-conditioned near the boundary.
This is measured explicitly by the repeated-cycle and boundary-stress tests; it
should later be controlled through bounded coupling outputs and a conditioning
regularizer, not rollback.
