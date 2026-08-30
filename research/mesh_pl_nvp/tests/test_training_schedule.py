from __future__ import annotations

import pytest

from research.mesh_pl_nvp.run_balls_patch import delayed_cosine_multiplier


def test_delayed_cosine_multiplier_holds_then_reaches_minimum() -> None:
    settings = {"total_steps": 1000, "decay_start": 300, "minimum_ratio": 1.0 / 30.0}
    assert delayed_cosine_multiplier(0, **settings) == 1.0
    assert delayed_cosine_multiplier(300, **settings) == 1.0
    assert delayed_cosine_multiplier(650, **settings) == pytest.approx(
        0.5 * (1.0 + settings["minimum_ratio"])
    )
    assert delayed_cosine_multiplier(1000, **settings) == pytest.approx(
        settings["minimum_ratio"]
    )
    assert delayed_cosine_multiplier(1200, **settings) == pytest.approx(
        settings["minimum_ratio"]
    )
