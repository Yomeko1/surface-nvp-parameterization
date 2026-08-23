from __future__ import annotations

import copy


class SafeCheckpoint:
    def __init__(self):
        self.model_state = None
        self.optimizer_state = None
        self.iteration = 0
        self.uv = None
        self.metrics = None

    def save(self, model, optimizer, iteration: int, uv, metrics: dict) -> None:
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        self.iteration = iteration
        self.uv = uv.copy()
        self.metrics = dict(metrics)

    def restore(self, model, optimizer) -> bool:
        if self.model_state is None:
            return False
        model.load_state_dict(self.model_state)
        optimizer.load_state_dict(self.optimizer_state)
        return True

    def restore_model(self, model) -> bool:
        if self.model_state is None:
            return False
        model.load_state_dict(self.model_state)
        return True
