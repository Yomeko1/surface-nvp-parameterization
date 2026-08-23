# v2.4 Results

This directory contains two distinct v2.4 experiment groups. They must not be
combined as though they used the same model or initialization.

## Tutte-initialized baseline

`Balls/`, `David328/`, `NefertitiFace/`, `Isis/`, `Cow/`, and `00027/` contain
seed-0 comparisons for Direct UV, Affine-NVP, Spline-NVP, and SLIM. NVP/direct
runs use 1000 iterations, SLIM uses 20 iterations, and all methods in a dataset
share the same geometry-scaled Tutte initialization.

| Dataset | Initial SD | Direct SD | Affine SD | Spline16 SD | SLIM SD | Invalid result |
|---|---:|---:|---:|---:|---:|---|
| Balls | 5.754598 | 4.413440 | 5.169155 | 4.502016 | 4.245895 | none |
| David328 | 28.409904 | 19.211613 | 10.462589 | 8.516175 | 7.237717 | none |
| NefertitiFace | 5.308797 | 4.041508 | 4.426826 | 4.058399 | 4.036175 | none |
| Isis | 13.462514 | 4.888492 | 7.622651 | 5.365119 | 4.530849 | none |
| Cow | 286.001272 | 286.001272 | 50.704037 | 12.807380 | 4.085300 | SLIM: 1069 intersections |
| 00027 | 978.423636 | 978.423636 | 112.891161 | 978.422440 | 4.018421 | none |

The low Cow SLIM SD is not a successful globally injective result. Direct and
Spline on 00027 select iteration 0 and expose the old non-cumulative rollback
learning-rate issue diagnosed in the release notes.

The baseline manifests were generated before a benchmark provenance fix. Their
`global_transform=false` and `spline_bins=null` fields are stale; each saved
run configuration records the effective `global_transform=true` and
`spline_bins=16`. The original manifests are intentionally preserved rather
than rewritten. New benchmark roots record the effective values correctly.

## Improved 1+2+3 experiments

`improved_123/eval/` contains a uniform 1000 Adam + 20 L-BFGS comparison on
four simple meshes. `improved_123/full/David328/` uses the complete 3000 Adam +
50 L-BFGS budget. See `improved_123/README.md` and
`configs/improved_123.yaml`.

The one-time implementation smoke run is intentionally omitted from the
release archive because it is superseded by the eval and full-budget runs.

## New initialization methods

Mean-value, ABF++, and validated `auto` initialization were implemented after
these Tutte baseline runs. Their six-mesh candidate results are documented in
the repository-level `v2.4.md`; they have not been used to overwrite this
archive. Use a new benchmark output root when measuring their downstream effect.
