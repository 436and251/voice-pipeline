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
class TrainingCursor:
    global_step: int = 0
    epoch: int = 0
    next_batch_index: int = 0


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / "training" / "s2" / "checkpoints" / f"step-{step:08d}.pt"


def save_checkpoint(
    path: Path,
    *,
    net_g,
    net_d,
    optim_g,
    optim_d,
    scheduler_g,
    scheduler_d,
    scaler,
    cursor: TrainingCursor,
) -> None:
    _validate_cursor(cursor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "format_version": FORMAT_VERSION,
        "profile": PROFILE,
        "global_step": cursor.global_step,
        "epoch": cursor.epoch,
        "next_batch_index": cursor.next_batch_index,
        "net_g": net_g.state_dict(),
        "net_d": net_d.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
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


def load_checkpoint(
    path: Path,
    *,
    net_g,
    net_d,
    optim_g,
    optim_d,
    scheduler_g,
    scheduler_d,
    scaler,
    target_optimizer_steps: int,
) -> TrainingCursor:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "profile",
        "global_step",
        "epoch",
        "next_batch_index",
        "net_g",
        "net_d",
        "optim_g",
        "optim_d",
        "scheduler_g",
        "scheduler_d",
        "scaler",
        "python_rng",
        "torch_rng",
        "cuda_rng",
    }
    if not isinstance(payload, dict):
        raise ValueError("invalid S2 checkpoint envelope")
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"invalid S2 checkpoint: missing {', '.join(sorted(missing))}")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported S2 checkpoint format_version")
    if payload["profile"] != PROFILE:
        raise ValueError("S2 checkpoint profile must be v2ProPlus")
    cursor = TrainingCursor(payload["global_step"], payload["epoch"], payload["next_batch_index"])
    _validate_cursor(cursor)
    match = _FILENAME.fullmatch(path.name)
    if match is None or int(match.group(1)) != cursor.global_step:
        raise ValueError("S2 checkpoint filename does not match embedded step")
    if cursor.global_step > target_optimizer_steps:
        raise ValueError("S2 checkpoint step exceeds requested target")

    _validate_model_state("net_g", payload["net_g"], net_g)
    _validate_model_state("net_d", payload["net_d"], net_d)
    _validate_optimizer_state("optim_g", payload["optim_g"], optim_g)
    _validate_optimizer_state("optim_d", payload["optim_d"], optim_d)
    _validate_structure("scheduler_g", payload["scheduler_g"], scheduler_g.state_dict())
    _validate_structure("scheduler_d", payload["scheduler_d"], scheduler_d.state_dict())
    _validate_structure("scaler", payload["scaler"], scaler.state_dict())
    _validate_rng(payload)

    net_g.load_state_dict(payload["net_g"], strict=True)
    net_d.load_state_dict(payload["net_d"], strict=True)
    optim_g.load_state_dict(payload["optim_g"])
    optim_d.load_state_dict(payload["optim_d"])
    scheduler_g.load_state_dict(payload["scheduler_g"])
    scheduler_d.load_state_dict(payload["scheduler_d"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return cursor


def _validate_cursor(cursor: TrainingCursor) -> None:
    values = (cursor.global_step, cursor.epoch, cursor.next_batch_index)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("invalid S2 training cursor")


def _validate_model_state(name: str, saved: object, model) -> None:
    current = model.state_dict()
    if not isinstance(saved, Mapping) or saved.keys() != current.keys():
        raise ValueError(f"invalid {name} state keys")
    for key, expected in current.items():
        value = saved[key]
        if isinstance(expected, torch.Tensor):
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape:
                raise ValueError(f"invalid {name} tensor: {key}")
        elif type(value) is not type(expected):
            raise ValueError(f"invalid {name} value: {key}")


def _validate_optimizer_state(name: str, saved: object, optimizer) -> None:
    if not isinstance(saved, Mapping) or set(saved) != {"state", "param_groups"}:
        raise ValueError(f"invalid {name} state")
    states = saved["state"]
    groups = saved["param_groups"]
    if not isinstance(states, Mapping) or not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError(f"invalid {name} parameter groups")
    current_groups = optimizer.state_dict()["param_groups"]
    parameters: dict[object, torch.Tensor] = {}
    for saved_group, live_group, current_group in zip(groups, optimizer.param_groups, current_groups):
        if not isinstance(saved_group, Mapping) or "params" not in saved_group:
            raise ValueError(f"invalid {name} parameter group")
        saved_ids = saved_group["params"]
        live_parameters = live_group["params"]
        if not isinstance(saved_ids, list) or len(saved_ids) != len(live_parameters):
            raise ValueError(f"invalid {name} parameter group size")
        if set(saved_group) != set(current_group):
            raise ValueError(f"invalid {name} parameter group keys")
        for saved_id, parameter in zip(saved_ids, live_parameters):
            try:
                duplicate = saved_id in parameters
            except TypeError as error:
                raise ValueError(f"invalid {name} parameter identifier") from error
            if duplicate:
                raise ValueError(f"duplicate {name} parameter identifier")
            parameters[saved_id] = parameter
    for saved_id, state in states.items():
        if saved_id not in parameters or not isinstance(state, Mapping):
            raise ValueError(f"invalid {name} parameter state")
        parameter = parameters[saved_id]
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step" and value.shape != parameter.shape:
                raise ValueError(f"invalid {name} tensor state: {key}")
            if key == "step" and (not isinstance(value, torch.Tensor) or value.numel() != 1):
                raise ValueError(f"invalid {name} step state")


def _validate_structure(name: str, saved: object, current: object) -> None:
    if isinstance(current, Mapping):
        if not isinstance(saved, Mapping) or saved.keys() != current.keys():
            raise ValueError(f"invalid {name} keys")
        for key in current:
            _validate_structure(f"{name}.{key}", saved[key], current[key])
    elif isinstance(current, (list, tuple)):
        if not isinstance(saved, type(current)) or len(saved) != len(current):
            raise ValueError(f"invalid {name} sequence")
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            _validate_structure(f"{name}[{index}]", saved_item, current_item)
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
        raise ValueError("invalid checkpoint RNG state") from error
    cuda_rng = payload["cuda_rng"]
    if torch.cuda.is_available():
        if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("invalid checkpoint CUDA RNG state")
        try:
            for index, state in enumerate(cuda_rng):
                torch.Generator(device=f"cuda:{index}").set_state(state)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint CUDA RNG state") from error
    elif cuda_rng is not None and not isinstance(cuda_rng, list):
        raise ValueError("invalid checkpoint CUDA RNG state")
