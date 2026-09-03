from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import S1TrainConfig


@dataclass(frozen=True, slots=True)
class S1MiniBatchResult:
    loss: float
    top3_accuracy: float


@dataclass(frozen=True, slots=True)
class S1OptimizerResult:
    learning_rate: float
    scaler_scale: float


def backward_s1_minibatch(batch, model, scaler, config: S1TrainConfig) -> S1MiniBatchResult:
    device = torch.device(config.device)
    values = {
        name: batch[name].to(device, non_blocking=device.type == "cuda")
        for name in (
            "phoneme_ids",
            "phoneme_ids_len",
            "semantic_ids",
            "semantic_ids_len",
            "bert_feature",
        )
    }
    fp16 = config.precision == "fp16" and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        loss, accuracy = model.forward_old(
            values["phoneme_ids"],
            values["phoneme_ids_len"],
            values["semantic_ids"],
            values["semantic_ids_len"],
            values["bert_feature"],
        )
    scaler.scale(loss).backward()
    return S1MiniBatchResult(float(loss.detach()), float(accuracy))


def finish_s1_optimizer_step(model, optimizer, scheduler, scaler) -> S1OptimizerResult:
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    learning_rate = float(optimizer.param_groups[0]["lr"])
    scheduler.step()
    optimizer.zero_grad()
    return S1OptimizerResult(learning_rate, float(scaler.get_scale()))


__all__ = [
    "S1MiniBatchResult",
    "S1OptimizerResult",
    "backward_s1_minibatch",
    "finish_s1_optimizer_step",
]
