from __future__ import annotations

from voice_pipeline.core.gpt_sovits.s1 import FixedS1LRSchedule, ScaledAdam

from .config import S1TrainConfig


def build_optimizer(model, config: S1TrainConfig) -> ScaledAdam:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("S1 model has no trainable parameters")
    return ScaledAdam(
        [parameter for _, parameter in named],
        lr=0.01,
        betas=(0.9, 0.95),
        clipping_scale=2.0,
        parameters_names=[[name for name, _ in named]],
        show_dominant_parameters=False,
        clipping_update_period=1000,
    )


def build_scheduler(optimizer: ScaledAdam) -> FixedS1LRSchedule:
    return FixedS1LRSchedule(optimizer)


__all__ = ["build_optimizer", "build_scheduler"]
