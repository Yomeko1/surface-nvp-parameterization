"""Conservative learning-rate recovery after radial conditioning risk subsides."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRecoveryEvent:
    """One accepted learning-rate recovery update."""

    step: int
    old_learning_rate: float
    new_learning_rate: float
    conditioning_risk_max: float


class RiskLearningRateRecovery:
    """Recover LR only after sustained conditioning-risk headroom.

    The controller is mesh independent: it observes only the normalized radial
    conditioning risk.  A threshold below the emergency reduction threshold
    supplies hysteresis, while patience and an update interval prevent rapid
    LR oscillation near the threshold.
    """

    def __init__(
        self,
        *,
        threshold: float,
        patience: int,
        interval: int,
        factor: float,
        maximum_learning_rate: float,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if interval < 1:
            raise ValueError("interval must be at least 1")
        if factor <= 1.0:
            raise ValueError("factor must be greater than 1")
        if maximum_learning_rate <= 0.0:
            raise ValueError("maximum_learning_rate must be positive")

        self.threshold = float(threshold)
        self.patience = int(patience)
        self.interval = int(interval)
        self.factor = float(factor)
        self.maximum_learning_rate = float(maximum_learning_rate)
        self._safe_steps = 0

    @property
    def safe_steps(self) -> int:
        return self._safe_steps

    def observe(
        self,
        *,
        step: int,
        risk_max: float,
        learning_rate: float,
    ) -> RiskRecoveryEvent | None:
        """Observe risk and optionally propose one bounded LR increase."""

        if risk_max > self.threshold:
            self._safe_steps = 0
            return None

        self._safe_steps += 1
        if self._safe_steps < self.patience:
            return None
        if (self._safe_steps - self.patience) % self.interval != 0:
            return None

        new_learning_rate = min(
            self.maximum_learning_rate,
            float(learning_rate) * self.factor,
        )
        if new_learning_rate <= learning_rate:
            return None
        return RiskRecoveryEvent(
            step=int(step),
            old_learning_rate=float(learning_rate),
            new_learning_rate=new_learning_rate,
            conditioning_risk_max=float(risk_max),
        )
