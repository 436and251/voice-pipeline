from __future__ import annotations

from collections.abc import Mapping

import torch


class FixedS1LRSchedule:
    """Preserve the pinned S1 scheduler's effective post-update LR behavior."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self._current_step = 0

    def step(self) -> float:
        for group in self.optimizer.param_groups:
            group["lr"] = 0.002
        self._current_step += 1
        return 0.002

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, int]:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state) != {"current_step"}
            or isinstance(state["current_step"], bool)
            or not isinstance(state["current_step"], int)
            or state["current_step"] < 0
        ):
            raise ValueError("invalid S1 scheduler state")
        self._current_step = state["current_step"]
        if self._current_step:
            for group in self.optimizer.param_groups:
                group["lr"] = 0.002


__all__ = ["FixedS1LRSchedule"]
