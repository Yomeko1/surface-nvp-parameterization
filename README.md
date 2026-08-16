# Surface NVP Parameterization

This project is an injectivity-aware surface parameterization research pipeline inspired by StructuredField's orientation-preserving NVP idea.

Goal:

```text
surface mesh -> initial valid UV -> 2D orientation-preserving NVP -> checked UV output
```

Version 2.0 supports a focused single-chart setting:

- One triangular mesh.
- Disk topology with one boundary loop.
- Tutte circle initialization.
- Affine Real-NVP or monotonic rational-quadratic spline deformation of the initial UV.
- A shared optimizer and objective for NVP and direct-UV experiments.
- An optional libigl SLIM baseline that starts from the same initial UV.
- Local flip checks by signed triangle area.
- Global UV self-intersection checks with rollback to the latest valid checkpoint.
- OBJ I/O by default, optional USD/USDA I/O when `pxr` is installed.

## Install

```bash
pip install -r requirements.txt
```

USD support is optional:

```bash
pip install usd-core
```

## Usage

Initialize UV:

```bash
python scripts/init_uv.py --input data/input/mesh.obj --output data/output/init.obj --method tutte
```

Train NVP:

```bash
python scripts/train_injective_nvp.py --input data/input/mesh.obj --output data/output/final.obj --iters 1000
```

Train the more expressive orientation-preserving spline NVP:

```bash
python scripts/train_injective_nvp.py --input data/input/mesh.obj --output data/output/spline.obj --coupling-type spline --spline-bins 8 --iters 1000
```

Train with a YAML config. Command-line arguments override values from the config:

```bash
python scripts/train_injective_nvp.py --config configs/default.yaml --input data/input/mesh.obj --output data/output/final.obj
```

Train the direct-UV baseline, which optimizes UV vertices directly without the NVP map:

```bash
python scripts/train_direct_uv.py --config configs/default.yaml --input data/input/mesh.obj --output data/output/direct.obj --iters 1000
```

Use exactly the same initial UV file for NVP and direct UV experiments:

```bash
python scripts/train_direct_uv.py --input data/input/mesh.obj --initial-uv data/input/shared_init.obj --output data/output/direct.obj
python scripts/train_injective_nvp.py --input data/input/mesh.obj --initial-uv data/input/shared_init.obj --output data/output/nvp.obj
```

The explicit initial-UV mesh must contain UV coordinates and have the same
vertex count and triangle topology as the input mesh. All optimization methods
reject an initial map with flips or global intersections.

Build the optional libigl SLIM baseline wrapper:

```bash
cmake -S external/slim_runner -B build/slim_runner
cmake --build build/slim_runner --config Release
```

Run SLIM from the same initial UV:

```bash
python scripts/run_slim.py --input data/input/mesh.obj --initial-uv data/input/shared_init.obj --output data/output/slim.obj --executable build/slim_runner/Release/surface_nvp_slim.exe --iters 20
```

If `--initial-uv` is omitted, all three methods use UVs already stored in the
input mesh, or the same deterministic Tutte initialization when no UV exists.
See `external/slim_runner/README.md` for build and license details.

Summarize multiple runs into a CSV/JSON table:

```bash
python scripts/summarize_metrics.py --inputs data/output/run_a/final.metrics.json data/output/run_b/direct.metrics.json --output data/output/summary.csv
```

Use CUDA for training and validation if PyTorch CUDA is available:

```bash
python scripts/train_injective_nvp.py --input data/input/mesh.obj --output data/output/final.obj --iters 1000 --device cuda
```

For large meshes, validation can be memory intensive because global triangle intersection checks consider triangle pairs in batches. Reduce the batch size if CUDA runs out of memory:

```bash
python scripts/train_injective_nvp.py --input data/input/mesh.obj --output data/output/final.obj --device cuda --intersection-batch-size 65536
```

Force CPU validation while still training on CUDA:

```bash
python scripts/train_injective_nvp.py --input data/input/mesh.obj --output data/output/final.obj --device cuda --validation-device cpu
```

Check UV:

```bash
python scripts/check_uv.py --input data/output/final.obj
```

## Running Your Own Files

OBJ example:

```bash
python scripts/init_uv.py --input F:/path/to/model.obj --output F:/path/to/init.obj
python scripts/train_injective_nvp.py --input F:/path/to/model.obj --output F:/path/to/final.obj --iters 1000 --device cuda
python scripts/check_uv.py --input F:/path/to/final.obj
```

USDA/USD example:

```bash
pip install usd-core
python scripts/init_uv.py --input F:/path/to/model.usda --output F:/path/to/init.usda
python scripts/train_injective_nvp.py --input F:/path/to/model.usda --output F:/path/to/final.usda --iters 1000 --device cuda
python scripts/check_uv.py --input F:/path/to/final.usda
```

If the USD file has multiple meshes, pass the mesh prim path:

```bash
python scripts/train_injective_nvp.py --input scene.usda --prim-path /World/MyMesh --output final.usda
```

## Input Mesh Requirements

This v2 prototype is intended for a simple parameterization setting:

- A single connected triangular mesh.
- Disk topology with one clear boundary loop.
- Non-degenerate triangles; avoid zero-area or nearly zero-area faces.
- Manifold-like connectivity; avoid duplicate faces, broken boundaries, or non-manifold edges.
- A valid initial UV is required for reliable optimization. If the input has no UV, the default Tutte initialization assumes the mesh has disk topology and a usable boundary.

Closed surfaces without cuts, meshes with multiple boundary loops, and heavily non-manifold meshes are outside the intended v2 scope. USD/USDA files with multiple mesh prims should use `--prim-path` to select the target mesh.

## Outputs

Training writes:

- `final.obj` or `final.usda`: mesh with optimized UV.
- `final.initial.uv.png`: initial UV triangle plot.
- `final.uv.png`: UV triangle plot.
- `final.compare.png`: side-by-side initial/final UV comparison.
- `final.initial.flip_heatmap.png`: initial UV signed-area heatmap for local flip inspection.
- `final.flip_heatmap.png`: final UV signed-area heatmap for local flip inspection.
- `final.area_compare.png`: initial/final signed-area heatmaps with a shared color scale.
- `final.distortion_compare.png`: initial/final per-face symmetric Dirichlet heatmaps with a shared clipped color scale.
- `final.loss.png`: training loss curves for valid checkpoints; invalid checkpoints are marked with red crosses.
- `final.config.json`: effective training config used for the run.
- `final.metrics.json`: validation history plus initial/final metrics.
- `final.summary.csv` and `final.summary.json`: compact per-run metric summary for reports.

The key metrics are:

- `num_flipped`: number of triangles with non-positive signed UV area. Target is `0`.
- `num_nonfinite`: number of NaN/Inf UV scalar coordinates. Target is `0`.
- `min_signed_area`: smallest signed UV triangle area. It should be positive.
- `num_intersections`: number of non-adjacent UV triangle intersections. Target is `0`.
- `is_valid`: true only when flips, non-finite coordinates, and intersections are all absent.

`final.metrics.json` also records distortion metrics for the initial and final UV:

- `symmetric_dirichlet_mean` and `symmetric_dirichlet_max`: conformal/isometric distortion summary. Lower is better.
- `symmetric_dirichlet_area_weighted_mean`: the training-aligned, 3D-face-area-weighted SD mean.
- `symmetric_dirichlet_median`, `p90`, `p95`, and `p99`: distribution statistics less brittle than a single maximum.
- `uv_area_min`, `uv_area_mean`, and `uv_area_max`: signed UV triangle area summary.
- `area_ratio_min`, `area_ratio_mean`, and `area_ratio_max`: absolute UV area divided by 3D triangle area.
- `edge_length_ratio_*`: UV edge length divided by 3D edge length. This is scale-dependent.
- `scaled_edge_length_ratio_*`: edge length ratios normalized by their median scale, useful for relative stretch.
- `angle_distortion_mean_deg` and `angle_distortion_max_deg`: per-triangle angle change in degrees.

Each history entry in `final.metrics.json` records the total loss and split terms:

- `loss`: weighted total training objective.
- `loss_distortion`: symmetric Dirichlet loss.
- `loss_boundary`, `loss_identity`, and `loss_jacobian`: raw auxiliary losses.
- `weighted_loss_boundary`, `weighted_loss_identity`, and `weighted_loss_jacobian`: weighted auxiliary losses actually contributing to `loss`.

`loss_area` and `weighted_loss_area` remain as compatibility aliases for the Jacobian barrier in metrics files.
The existing `area_weight` configuration key now weights this scale-normalized Jacobian barrier; its name is retained so older YAML configurations continue to work.

NVP and direct-UV training both select the best valid checkpoint by `loss_distortion` for final output. The chosen checkpoint is recorded in `final.metrics.json` under `training.selected_iteration`.

For visual inspection, open `final.uv.png`. A valid result should not show overlapped non-adjacent triangles. For OBJ, any DCC/viewer that displays UVs can also inspect `vt` coordinates.

For local flip inspection, open `final.initial.flip_heatmap.png` and `final.flip_heatmap.png`:

- Red means negative signed area, i.e. a flipped UV triangle.
- Yellow means close to zero signed area, i.e. nearly collapsed or risky.
- Green means safely positive signed area.

For before/after comparison, prefer `final.area_compare.png` because it uses the same signed-area color scale for the initial and final UV. Use `final.distortion_compare.png` to see where symmetric Dirichlet distortion is high before and after NVP optimization.

When writing OBJ from an OBJ input, the writer preserves common display data where possible: `mtllib`, `usemtl`, and per-vertex `vn` normals. The geometry vertices and face topology are unchanged; only UV coordinates are rewritten.

## Project Layout

```text
surface_nvp/
  io/              mesh data, OBJ and optional USD readers/writers
  geometry/        topology and boundary extraction
  init_param/      shared initial-UV loading and Tutte initialization
  models/          direct UV, affine coupling, and spline coupling models
  losses/          weighted distortion, Jacobian barrier, and regularization
  injectivity/     signed area and triangle intersection validation
  training/        shared config, trainer, metrics, summaries, and rollback
  visualization/   simple UV plotting/export helpers
scripts/
  init_uv.py                 generate a Tutte initialization
  train_direct_uv.py         direct-vertex optimization baseline
  train_injective_nvp.py     affine or spline NVP optimization
  run_slim.py                optional external SLIM baseline
  check_uv.py                validate an output UV map
  summarize_metrics.py       aggregate run metrics
external/slim_runner/        minimal libigl SLIM command-line wrapper
tests/                       loss, initialization, trainer, and NVP tests
```

## Important Notes

The NVP map is orientation-preserving in continuous 2D parameter space because each coupling layer has positive Jacobian determinant. In the discrete mesh, we still validate triangle signed areas and triangle intersections because vertices are mapped and then connected by straight UV edges.

The method preserves injectivity best when the initial UV is already valid. Tutte circle initialization is the default because it is the safest first choice for a disk-topology triangular mesh.

The checked outputs currently stored in `data/output/` are the formal v1 affine-NVP and direct-UV runs for Balls, David328, NefertitiFace, Cow, and Isis. They are retained as historical baselines; formal v2 affine/spline/SLIM comparisons have not yet been run.

See `v2.md` for the exact v2.0 snapshot status, verification record, and known limitations. Git tags `v1` and `v2.0` identify the two rollback points.
