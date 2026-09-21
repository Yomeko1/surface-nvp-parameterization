# Mesh-aligned PL-NVP v3.2

The released default is **H1 affine100 -> Local RQS500 -> wide affine H2 300**.
Only H2's bounds change to `max_log_scale=0.16, max_shift=0.40`; H1 keeps
`0.08 / 0.20`. Six-ring scaffold, C0 boundary hats, float64, regularized SD and
risk-adaptive learning-rate reduction/recovery retain the v3.1 configuration.
No rollback, flip barrier or intersection penalty is used.

See [PRINCIPLE_CN.md](PRINCIPLE_CN.md) for the structural argument,
[RUN_PIPELINE_CN.md](RUN_PIPELINE_CN.md) for commands,
[AUDIT_PROTOCOL_CN.md](AUDIT_PROTOCOL_CN.md) for metric definitions and
[../../v3.2.md](../../v3.2.md) for results and limitations.

## Run

```bash
python -m pip install -e ".[test,usd]"
python -m research.mesh_pl_nvp.run_v32 --config research/mesh_pl_nvp/v3_2_default.yaml --input data/input/Cow/Cow_dABF.usda --output-dir data/output/v3.2/Cow
```

`surface-nvp-pl` is the installed console command. The YAML is optional: built-in
v3.2 defaults are identical. Explicit CLI options override YAML. Use `--device cpu`
without CUDA. The runner creates a new output directory and refuses to overwrite.
OBJ input defaults to OBJ output; USD input defaults to USDA. `--export-format usda`
reproduces the historical export-quantization audit and requires `usd-core`.

For a short smoke run add `--harmonic-iters 2 --local-iters 2 --final-harmonic-iters 2`.
This is an installation check, not a quality benchmark.

## Results and Audits

`final.obj`/`final.usda`, `final.model.pt` and native snapshots are saved before
network inverse auditing. `audit/` holds four distinct metrics and explicit
failure statuses; `report/` holds a JSON table, Markdown report and plots.
Original and extended geometry are checked separately, including quantized export.
An inverse status of `ok` only means finite computation; inspect the residual.
00027 still exhibits composed-network inverse failure despite valid geometric UV.

| Mesh | v3.1 SD | v3.2 SD | Improvement |
|---|---:|---:|---:|
| 00027 | 54.434032 | 47.082369 | 13.51% |
| Cow | 15.062480 | 13.992588 | 7.10% |

Native-f64 original-3D-area-weighted regularized SD, seed 20260906. Both adopted
endpoints have zero audited flips and intersections. These tests do not prove
universal improvement. See [benchmarks/v3_2_results.json](benchmarks/v3_2_results.json).

## Retained Entry Points

- v3.1: `run_harmonic_local.py` with unchanged `v3_1_default.yaml`, H100/Local500/H100.
- v3.0: `run_pipeline.py` with `default.yaml`, single-stage affine baseline.
- Historical/shared audit backend: `run_v31_audit.py` and `summarize_v31_audit.py`.
- v3.2 user entry point: `run_v32.py`, installed as `surface-nvp-pl`.

Do not load the v3.1 YAML through the v3.2 entry point to reproduce the old model:
its unlisted H2 settings would inherit v3.2 defaults. Use the v3.1 entry point or tag.
Historical exact reproduction should use the corresponding tag and dependencies.

## Tests

```bash
python -m pytest tests research/mesh_pl_nvp/tests -q
```

Coverage includes bounded pair enumeration, geometry/metric normalization,
H2-only bounds, v3.1 configuration compatibility, inverse-failure persistence,
portable model reloading and re-auditing, in addition to the retained PL layer,
scaffold and baseline tests. Large output/checkpoint files stay local.
