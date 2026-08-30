from argparse import Namespace

import pytest

from research.mesh_pl_nvp.pipeline_config import (
    apply_cli_overrides,
    load_pipeline_config,
)


def test_default_research_pipeline_configuration_is_current_small_run() -> None:
    config = load_pipeline_config(None)
    assert config["model"]["cycles"] == 4
    assert config["train"]["iters"] == 1000
    assert config["train"]["lr_schedule"] == "adaptive-plateau"
    assert config["scaffold"] == {"enabled": True, "scale": 1.1}


def test_cli_overrides_do_not_mutate_source_configuration() -> None:
    config = load_pipeline_config(None)
    args = Namespace(iters=12, scaffold=False)
    updated = apply_cli_overrides(config, args)
    assert updated["train"]["iters"] == 12
    assert updated["scaffold"]["enabled"] is False
    assert config["train"]["iters"] == 1000
    assert config["scaffold"]["enabled"] is True


def test_invalid_scaffold_scale_is_rejected() -> None:
    config = load_pipeline_config(None)
    args = Namespace(scaffold_scale=1.0)
    with pytest.raises(ValueError, match="greater than one"):
        apply_cli_overrides(config, args)
