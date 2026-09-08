from __future__ import annotations

import pytest

from research.mesh_pl_nvp.risk_recovery import RiskLearningRateRecovery


def _controller() -> RiskLearningRateRecovery:
    return RiskLearningRateRecovery(
        threshold=0.94,
        patience=3,
        interval=2,
        factor=1.1,
        maximum_learning_rate=2.0e-4,
    )


def test_recovery_waits_for_sustained_safe_risk_and_uses_interval() -> None:
    controller = _controller()
    learning_rate = 1.0e-4
    events = []
    for step in range(1, 8):
        event = controller.observe(
            step=step,
            risk_max=0.92,
            learning_rate=learning_rate,
        )
        if event is not None:
            learning_rate = event.new_learning_rate
            events.append(event)

    assert [event.step for event in events] == [3, 5, 7]
    assert learning_rate == pytest.approx(1.331e-4)


def test_risk_in_hysteresis_band_resets_recovery_patience() -> None:
    controller = _controller()
    assert controller.observe(step=1, risk_max=0.92, learning_rate=1.0e-4) is None
    assert controller.observe(step=2, risk_max=0.945, learning_rate=1.0e-4) is None
    assert controller.safe_steps == 0
    assert controller.observe(step=3, risk_max=0.92, learning_rate=1.0e-4) is None
    assert controller.observe(step=4, risk_max=0.92, learning_rate=1.0e-4) is None
    event = controller.observe(step=5, risk_max=0.92, learning_rate=1.0e-4)
    assert event is not None
    assert event.step == 5


def test_recovery_never_exceeds_safe_peak() -> None:
    controller = RiskLearningRateRecovery(
        threshold=0.94,
        patience=1,
        interval=1,
        factor=2.0,
        maximum_learning_rate=1.5e-4,
    )
    event = controller.observe(step=1, risk_max=0.90, learning_rate=1.0e-4)
    assert event is not None
    assert event.new_learning_rate == pytest.approx(1.5e-4)
    assert (
        controller.observe(
            step=2,
            risk_max=0.90,
            learning_rate=event.new_learning_rate,
        )
        is None
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("threshold", 1.0), ("patience", 0), ("interval", 0), ("factor", 1.0)),
)
def test_invalid_recovery_configuration_is_rejected(keyword: str, value: float) -> None:
    config = {
        "threshold": 0.94,
        "patience": 3,
        "interval": 2,
        "factor": 1.1,
        "maximum_learning_rate": 2.0e-4,
    }
    config[keyword] = value
    with pytest.raises(ValueError):
        RiskLearningRateRecovery(**config)
