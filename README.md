# Surface NVP Parameterization

This project is a first Python version of an injectivity-oriented surface parameterization pipeline inspired by StructuredField's orientation-preserving NVP idea.

Goal:

```text
surface mesh -> initial valid UV -> 2D orientation-preserving NVP -> checked UV output
```

The first version intentionally supports a simple setting:

- One triangular mesh.
- Disk topology with one boundary loop.
- Tutte circle initialization.
- 2D Real-NVP-style deformation of the initial UV.
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

Train with a YAML config. Command-line arguments override values from the config:

```bash
python scripts/train_injective_nvp.py --config configs/default.yaml --input data/input/mesh.obj --output data/output/final.obj
```

Train the direct-UV baseline, which optimizes UV vertices directly without the NVP map:

```bash
python scripts/train_direct_uv.py --config configs/default.yaml --input data/input/mesh.obj --output data/output/direct.obj --iters 1000
```

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

This v1 prototype is intended for a simple parameterization setting:

- A single connected triangular mesh.
- Disk topology with one clear boundary loop.
- Non-degenerate triangles; avoid zero-area or nearly zero-area faces.
- Manifold-like connectivity; avoid duplicate faces, broken boundaries, or non-manifold edges.
- A valid initial UV is required for reliable optimization. If the input has no UV, the default Tutte initialization assumes the mesh has disk topology and a usable boundary.

Closed surfaces without cuts, meshes with multiple boundary loops, and heavily non-manifold meshes are outside the intended v1 scope. USD/USDA files with multiple mesh prims should use `--prim-path` to select the target mesh.

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
- `min_signed_area`: smallest signed UV triangle area. It should be positive.
- `num_intersections`: number of non-adjacent UV triangle intersections. Target is `0`.
- `is_valid`: true only when `num_flipped == 0` and `num_intersections == 0`.

`final.metrics.json` also records distortion metrics for the initial and final UV:

- `symmetric_dirichlet_mean` and `symmetric_dirichlet_max`: conformal/isometric distortion summary. Lower is better.
- `uv_area_min`, `uv_area_mean`, and `uv_area_max`: signed UV triangle area summary.
- `area_ratio_min`, `area_ratio_mean`, and `area_ratio_max`: absolute UV area divided by 3D triangle area.
- `edge_length_ratio_*`: UV edge length divided by 3D edge length. This is scale-dependent.
- `scaled_edge_length_ratio_*`: edge length ratios normalized by their median scale, useful for relative stretch.
- `angle_distortion_mean_deg` and `angle_distortion_max_deg`: per-triangle angle change in degrees.

Each history entry in `final.metrics.json` records the total loss and split terms:

- `loss`: weighted total training objective.
- `loss_distortion`: symmetric Dirichlet loss.
- `loss_boundary`, `loss_identity`, and `loss_area`: raw auxiliary losses.
- `weighted_loss_boundary`, `weighted_loss_identity`, and `weighted_loss_area`: weighted auxiliary losses actually contributing to `loss`.

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
  geometry/        topology, boundary extraction, cotangent/local geometry
  init_param/      boundary mapping and Tutte initialization
  models/          2D NVP model
  losses/          distortion, area barrier, boundary/regularization losses
  injectivity/     signed area and triangle intersection validation
  training/        trainer, metrics, rollback checkpoint logic
  visualization/   simple UV plotting/export helpers
```

## Important Notes

The NVP map is orientation-preserving in continuous 2D parameter space because each coupling layer has positive Jacobian determinant. In the discrete mesh, we still validate triangle signed areas and triangle intersections because vertices are mapped and then connected by straight UV edges.

The method preserves injectivity best when the initial UV is already valid. Tutte circle initialization is the default because it is the safest first choice for a disk-topology triangular mesh.

The checked v1 outputs in `data/output/` include NVP and direct-UV baseline runs for Balls, David328, NefertitiFace, Cow, and Isis. Each formal run records metrics, summaries, loss plots, UV plots, flip heatmaps, area comparisons, and distortion comparisons.
