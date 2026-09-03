from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
import uuid

import torch


FORMAT_VERSION = 1
PROFILE = "v2ProPlus"
_FILENAME = re.compile(r"step-(\d{8})\.pt")


@dataclass(frozen=True, slots=True)
class S1TrainingCursor:
    optimizer_step: int = 0
    epoch: int = 0
    next_batch_index: int = 0
    accumulation_position: int = 0


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / "training" / "s1" / "checkpoints" / f"step-{step:08d}.pt"


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, cursor: S1TrainingCursor) -> None:
    _validate_cursor(cursor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "format_version": FORMAT_VERSION,
        "profile": PROFILE,
        "optimizer_step": cursor.optimizer_step,
        "epoch": cursor.epoch,
        "next_batch_index": cursor.next_batch_index,
        "accumulation_position": cursor.accumulation_position,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, target_optimizer_steps: int) -> S1TrainingCursor:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version", "profile", "optimizer_step", "epoch", "next_batch_index",
        "accumulation_position", "model", "optimizer", "scheduler", "scaler",
        "python_rng", "torch_rng", "cuda_rng",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid S1 checkpoint envelope")
    if payload["format_version"] != FORMAT_VERSION or payload["profile"] != PROFILE:
        raise ValueError("unsupported S1 checkpoint version or profile")
    cursor = S1TrainingCursor(
        payload["optimizer_step"], payload["epoch"], payload["next_batch_index"], payload["accumulation_position"]
    )
    _validate_cursor(cursor)
    match = _FILENAME.fullmatch(path.name)
    if match is None or int(match.group(1)) != cursor.optimizer_step:
        raise ValueError("S1 checkpoint filename does not match embedded step")
    if cursor.optimizer_step > target_optimizer_steps:
        raise ValueError("S1 checkpoint step exceeds requested target")
    _validate_model(payload["model"], model)
    _validate_optimizer(payload["optimizer"], optimizer)
    _validate_scheduler(payload["scheduler"])
    _validate_structure("scaler", payload["scaler"], scaler.state_dict())
    _validate_rng(payload)

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return cursor


def _validate_cursor(cursor: S1TrainingCursor) -> None:
    values = (cursor.optimizer_step, cursor.epoch, cursor.next_batch_index, cursor.accumulation_position)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("invalid S1 training cursor")
    if cursor.accumulation_position != 0:
        raise ValueError("S1 checkpoint accumulation_position must be zero")


def _validate_model(saved: object, model) -> None:
    current = model.state_dict()
    if not isinstance(saved, Mapping) or saved.keys() != current.keys():
        raise ValueError("invalid S1 model state keys")
    for key, expected in current.items():
        value = saved[key]
        if isinstance(expected, torch.Tensor) and (not isinstance(value, torch.Tensor) or value.shape != expected.shape):
            raise ValueError(f"invalid S1 model tensor: {key}")


def _validate_optimizer(saved: object, optimizer) -> None:
    if not isinstance(saved, Mapping) or set(saved) != {"state", "param_groups"}:
        raise ValueError("invalid S1 optimizer state")
    groups = saved["param_groups"]
    states = saved["state"]
    current_groups = optimizer.state_dict()["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(current_groups) or not isinstance(states, Mapping):
        raise ValueError("invalid S1 optimizer groups")
    parameter_ids = set()
    for saved_group, current_group in zip(groups, current_groups):
        if not isinstance(saved_group, Mapping) or set(saved_group) != set(current_group):
            raise ValueError("invalid S1 optimizer group keys")
        if not isinstance(saved_group["params"], list) or len(saved_group["params"]) != len(current_group["params"]):
            raise ValueError("invalid S1 optimizer group size")
        parameter_ids.update(saved_group["params"])
    if any(identifier not in parameter_ids or not isinstance(state, Mapping) for identifier, state in states.items()):
        raise ValueError("invalid S1 optimizer parameter state")
    for state in states.values():
        if "step" in state and (isinstance(state["step"], bool) or not isinstance(state["step"], int) or state["step"] < 0):
            raise ValueError("invalid S1 optimizer step state")
        if any(not torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)):
            raise ValueError("invalid S1 optimizer tensor state")


def _validate_scheduler(saved: object) -> None:
    if (
        not isinstance(saved, Mapping) or set(saved) != {"current_step"}
        or isinstance(saved["current_step"], bool) or not isinstance(saved["current_step"], int)
        or saved["current_step"] < 0
    ):
        raise ValueError("invalid S1 scheduler state")


def _validate_structure(name: str, saved: object, current: object) -> None:
    if isinstance(current, Mapping):
        if not isinstance(saved, Mapping) or saved.keys() != current.keys():
            raise ValueError(f"invalid {name} keys")
        for key in current:
            _validate_structure(f"{name}.{key}", saved[key], current[key])
    elif isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape:
            raise ValueError(f"invalid {name} tensor")
    elif type(saved) is not type(current):
        raise ValueError(f"invalid {name} value")


def _validate_rng(payload: Mapping[str, object]) -> None:
    try:
        random.Random().setstate(payload["python_rng"])
        torch.Generator().set_state(payload["torch_rng"])
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("invalid S1 checkpoint RNG state") from error
    cuda_rng = payload["cuda_rng"]
    if torch.cuda.is_available():
        if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("invalid S1 checkpoint CUDA RNG state")
    elif cuda_rng is not None and not isinstance(cuda_rng, list):
        raise ValueError("invalid S1 checkpoint CUDA RNG state")


__all__ = ["S1TrainingCursor", "checkpoint_path", "load_checkpoint", "save_checkpoint"]
