"""Mesh-independent relative-plateau learning-rate control."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


@dataclass(frozen=True)
class LearningRateEvent:
    step: int
    old_learning_rate: float
    new_learning_rate: float
    reason: str
    relative_improvement: float
    recent_q_p95: float
    recent_area_ratio: float


class AdaptivePlateauController:
    """Reduce LR after repeated weak relative improvement or sustained risk.

    Decisions are made only once per full window. Loss thresholds are relative,
    so no mesh-specific absolute SD scale or fixed decay iteration is required.
    """

    def __init__(
        self,
        *,
        initial_learning_rate: float,
        minimum_learning_rate: float,
        initial_minimum_area: float,
        window: int = 100,
        patience: int = 2,
        relative_threshold: float = 8.0e-3,
        factor: float = 0.5,
        q_threshold: float = 0.97,
        minimum_area_ratio: float = 0.25,
    ) -> None:
        if window <= 0 or patience <= 0:
            raise ValueError("window and patience must be positive")
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must lie in (0, 1)")
        if not 0.0 < minimum_learning_rate <= initial_learning_rate:
            raise ValueError("invalid minimum learning rate")
        if initial_minimum_area <= 0.0:
            raise ValueError("initial minimum area must be positive")
        self.learning_rate = float(initial_learning_rate)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.initial_minimum_area = float(initial_minimum_area)
        self.window = int(window)
        self.patience = int(patience)
        self.relative_threshold = float(relative_threshold)
        self.factor = float(factor)
        self.q_threshold = float(q_threshold)
        self.minimum_area_ratio = float(minimum_area_ratio)
        self.losses: list[float] = []
        self.q_values: list[float] = []
        self.minimum_areas: list[float] = []
        self.stale_windows = 0
        self.last_relative_improvement: float | None = None

    def observe(
        self,
        *,
        step: int,
        loss: float,
        q_max: float,
        minimum_area: float,
    ) -> LearningRateEvent | None:
        self.losses.append(float(loss))
        self.q_values.append(float(q_max))
        self.minimum_areas.append(float(minimum_area))
        if step <= 0 or step % self.window != 0 or len(self.losses) < 2 * self.window:
            return None

        old_loss = median(self.losses[-2 * self.window : -self.window])
        recent_loss = median(self.losses[-self.window :])
        relative = (old_loss - recent_loss) / max(abs(old_loss), 1.0e-15)
        self.last_relative_improvement = relative
        if relative < self.relative_threshold:
            self.stale_windows += 1
        else:
            self.stale_windows = 0

        recent_q_p95 = _quantile(self.q_values[-self.window :], 0.95)
        recent_area_ratio = median(self.minimum_areas[-self.window :]) / self.initial_minimum_area
        safety_reason = None
        if recent_q_p95 >= self.q_threshold:
            safety_reason = "q_p95"
        elif recent_area_ratio <= self.minimum_area_ratio:
            safety_reason = "minimum_area"
        plateau = self.stale_windows >= self.patience
        if not plateau and safety_reason is None:
            return None

        new_learning_rate = max(
            self.minimum_learning_rate, self.factor * self.learning_rate
        )
        if new_learning_rate >= self.learning_rate:
            return None
        event = LearningRateEvent(
            step=step,
            old_learning_rate=self.learning_rate,
            new_learning_rate=new_learning_rate,
            reason=safety_reason or "relative_plateau",
            relative_improvement=relative,
            recent_q_p95=recent_q_p95,
            recent_area_ratio=recent_area_ratio,
        )
        self.learning_rate = new_learning_rate
        self.stale_windows = 0
        return event
