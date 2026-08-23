# Spline 1+2+3 improvement experiment

This experiment combines three changes while preserving the existing v2.4 baseline outputs:

1. Cumulative rollback learning-rate decay, cosine Adam scheduling, and an optional L-BFGS phase.
2. Learnable orientation-preserving rotation mixing between coupling layers.
3. A larger Spline-NVP: 12 coupling layers, hidden dimension 128, and 32 spline bins.

The reusable configuration is `configs/improved_123.yaml` (3000 Adam + 50 L-BFGS).  The initial evaluation under `eval/` uses a reduced, uniform budget of 1000 Adam + 20 L-BFGS iterations so all four simple meshes can be compared at manageable cost.  All runs use seed 0, CUDA training/validation, the existing shared v2.4 initial UV, and `global_transform=true`.

| Dataset | v2.4 Spline16 SD | Improved SD | Relative change | SLIM SD | Improved gap to SLIM | Valid | Training seconds |
|---|---:|---:|---:|---:|---:|---|---:|
| Balls | 4.502016 | 4.483172 | -0.42% | 4.245895 | +5.59% | yes | 208.9 |
| David328 | 8.516175 | 8.163346 | -4.14% | 7.237717 | +12.79% | yes | 211.6 |
| NefertitiFace | 4.058399 | 4.068797 | +0.26% | 4.036175 | +0.81% | yes | 204.1 |
| Isis | 5.365119 | 5.270774 | -1.76% | 4.530849 | +16.33% | yes | 213.9 |

All four runs have zero flips, zero non-finite values, zero global intersections, and zero rollbacks.  In every run the selected checkpoint is Adam iteration 1000; the 20 L-BFGS steps remain valid but do not improve the selected distortion.  The larger model therefore helps the harder David328 and Isis cases, helps Balls only slightly, and mildly regresses the already near-optimal NefertitiFace case.  It also costs roughly twice the training time of the v2.4 Spline16 baseline.

The separate NefertitiFace 100 Adam + 2 L-BFGS implementation smoke artifact
was removed from the release archive after the eval and full-budget runs
superseded it.

## Full planned-budget check

`full/David328/` uses the complete configuration budget of 3000 Adam + 50 L-BFGS iterations.  It reaches area-weighted SD `7.877279`, a 7.50% improvement over the v2.4 Spline16 result and an 8.84% gap to SLIM.  SD P95 improves from `21.6573` to `18.0584`, and the area-weighted condition number improves from `2.8572` to `2.6843`.

The run has one invalid Adam checkpoint at iteration 700.  Cumulative rollback decay reduces the effective learning rate and training subsequently remains valid through iteration 3050.  The selected checkpoint is Adam iteration 3000; as in the reduced-budget runs, L-BFGS remains valid but does not improve the selected loss.  Total training time is 625.7 seconds on the recorded RTX 3070 Ti Laptop GPU environment.
