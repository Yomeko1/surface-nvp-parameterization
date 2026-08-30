from __future__ import annotations

from research.mesh_pl_nvp.adaptive_lr import AdaptivePlateauController


def _run(loss_scale: float) -> list[tuple[int, float]]:
    controller = AdaptivePlateauController(
        initial_learning_rate=3.0e-3,
        minimum_learning_rate=1.0e-5,
        initial_minimum_area=0.1,
        window=20,
        patience=2,
        relative_threshold=2.0e-3,
        factor=0.3,
    )
    events = []
    for step in range(161):
        loss = loss_scale * (2.0 - 0.004 * min(step, 80))
        event = controller.observe(step=step, loss=loss, q_max=0.7, minimum_area=0.1)
        if event is not None:
            events.append((event.step, event.new_learning_rate))
    return events


def test_relative_plateau_is_scale_independent() -> None:
    events = _run(1.0)
    assert events
    assert events == _run(1000.0)
    assert events[0][0] > 80
    assert events[0][1] == 9.0e-4


def test_sustained_q_risk_can_reduce_before_loss_plateau() -> None:
    controller = AdaptivePlateauController(
        initial_learning_rate=1.0e-3,
        minimum_learning_rate=1.0e-5,
        initial_minimum_area=0.1,
        window=10,
        patience=3,
        relative_threshold=1.0e-6,
        q_threshold=0.95,
    )
    event = None
    for step in range(21):
        event = controller.observe(
            step=step,
            loss=2.0 - 0.01 * step,
            q_max=0.98,
            minimum_area=0.1,
        )
    assert event is not None
    assert event.reason == "q_p95"
