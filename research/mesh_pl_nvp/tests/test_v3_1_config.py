from __future__ import annotations

from pathlib import Path

import pytest

from research.mesh_pl_nvp.run_harmonic_local import parse_args


def test_release_config_matches_v3_1_defaults() -> None:
    root = Path(__file__).resolve().parents[3]
    config = root / "research" / "mesh_pl_nvp" / "v3_1_default.yaml"
    args = parse_args(
        [
            "--config",
            str(config),
            "--input",
            "mesh.usda",
            "--output-dir",
            "output",
        ]
    )
    assert args.harmonic_iters == 100
    assert args.local_iters == 500
    assert args.final_harmonic_iters == 100
    assert args.local_latent_transform == "spline"
    assert args.local_radial_map == "atanh"
    assert args.scaffold_rings == 6
    assert args.boundary_hat_counts == [4, 8, 16, 32]
    assert args.auto_boundary_hat_counts
    assert args.local_risk_adaptive
    assert args.local_risk_recovery

    built_in = parse_args(["--input", "mesh.usda", "--output-dir", "output"])
    ignored = {"config", "input", "output_dir", "prim_path"}
    for key, value in vars(built_in).items():
        if key not in ignored:
            assert getattr(args, key) == value, key


def test_cli_overrides_release_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("local_iters: 400\ndevice: cpu\n", encoding="utf-8")
    args = parse_args(
        [
            "--config",
            str(config),
            "--input",
            "mesh.usda",
            "--output-dir",
            "output",
            "--local-iters",
            "50",
            "--no-local-risk-recovery",
        ]
    )
    assert args.local_iters == 50
    assert args.device == "cpu"
    assert not args.local_risk_recovery


def test_unknown_release_config_key_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("cow_only_magic: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown v3.1 config keys"):
        parse_args(
            [
                "--config",
                str(config),
                "--input",
                "mesh.usda",
                "--output-dir",
                "output",
            ]
        )


@pytest.mark.parametrize(
    "contents",
    (
        "local_latent_transform: cubic\n",
        "local_risk_recovery: yes-please\n",
        "frequencies: 2\n",
    ),
)
def test_invalid_release_config_value_is_rejected(
    tmp_path: Path,
    contents: str,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_args(
            [
                "--config",
                str(config),
                "--input",
                "mesh.usda",
                "--output-dir",
                "output",
            ]
        )
