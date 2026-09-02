from __future__ import annotations

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
