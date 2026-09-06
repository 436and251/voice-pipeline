from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import torch

from voice_pipeline.profiles.registry import ProfileRegistry

from .config import EvaluationConfig


_FILENAME = re.compile(r"step-(\d{8})\.pt")


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    kind: str
    path: Path
    step: int | None
    sha256: str
    is_base: bool


@dataclass(frozen=True, slots=True)
class CandidatePair:
    key: str
    s1: CheckpointRef
    s2: CheckpointRef


def discover_checkpoints(config: EvaluationConfig) -> tuple[tuple[CheckpointRef, ...], tuple[CheckpointRef, ...]]:
    profile = ProfileRegistry.get("v2ProPlus")
    base_s1 = _base_ref("s1", config.project_root / profile.s1_relative_path)
    base_s2 = _base_ref("s2", config.project_root / profile.s2g_relative_path)
    s1 = (base_s1, *discover_s1_checkpoints(config.run_dir / "training" / "s1" / "checkpoints"))
    s2 = (base_s2, *discover_s2_checkpoints(config.run_dir / "training" / "s2" / "checkpoints"))
    return s1, s2


def discover_s1_checkpoints(directory: Path) -> tuple[CheckpointRef, ...]:
    return _discover(directory, "s1", "optimizer_step", "model")


def discover_s2_checkpoints(directory: Path) -> tuple[CheckpointRef, ...]:
    return _discover(directory, "s2", "global_step", "net_g")


def stage_one_pairs(base_s1: CheckpointRef, s2_refs: tuple[CheckpointRef, ...]) -> tuple[CandidatePair, ...]:
    _kind(base_s1, "s1")
    if not base_s1.is_base:
        raise ValueError("stage one requires the base S1 checkpoint")
    return tuple(_pair(base_s1, s2) for s2 in s2_refs)


def stage_two_pairs(
    s1_refs: tuple[CheckpointRef, ...],
    retained_s2: tuple[CheckpointRef, ...],
) -> tuple[CandidatePair, ...]:
    return tuple(_pair(s1, s2) for s2 in retained_s2 for s1 in s1_refs)


def _discover(directory: Path, kind: str, step_field: str, weights_field: str) -> tuple[CheckpointRef, ...]:
    directory = Path(directory)
    if not directory.is_dir():
        return ()
    refs: list[CheckpointRef] = []
    seen: set[int] = set()
    for path in sorted(directory.glob("step-*.pt")):
        match = _FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid {kind.upper()} checkpoint filename: {path.name}")
        payload = _load(path, kind)
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported {kind.upper()} checkpoint format_version")
        if payload.get("profile") != "v2ProPlus":
            raise ValueError(f"{kind.upper()} checkpoint profile must be v2ProPlus")
        step = payload.get(step_field)
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"invalid {kind.upper()} optimizer step")
        if int(match.group(1)) != step:
            raise ValueError(f"{kind.upper()} checkpoint filename does not match embedded step")
        weights = payload.get(weights_field)
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError(f"invalid {kind.upper()} checkpoint weights")
        if step in seen:
            raise ValueError(f"duplicate {kind.upper()} checkpoint step: {step}")
        seen.add(step)
        refs.append(CheckpointRef(kind, path.resolve(), step, _sha256(path), False))
    return tuple(sorted(refs, key=lambda ref: ref.step))


def _load(path: Path, kind: str) -> Mapping:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"invalid {kind.upper()} checkpoint {path.name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid {kind.upper()} checkpoint envelope")
    return payload


def _base_ref(kind: str, path: Path) -> CheckpointRef:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"base {kind.upper()} checkpoint does not exist: {path}")
    return CheckpointRef(kind, path, None, _sha256(path), True)


def _pair(s1: CheckpointRef, s2: CheckpointRef) -> CandidatePair:
    _kind(s1, "s1")
    _kind(s2, "s2")
    s1_label = "base" if s1.is_base else f"{s1.step:08d}-{s1.sha256[:12]}"
    s2_label = "base" if s2.is_base else f"{s2.step:08d}-{s2.sha256[:12]}"
    return CandidatePair(f"s1-{s1_label}--s2-{s2_label}", s1, s2)


def _kind(ref: CheckpointRef, expected: str) -> None:
    if ref.kind != expected:
        raise ValueError(f"expected {expected.upper()} checkpoint, got {ref.kind}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CandidatePair",
    "CheckpointRef",
    "discover_checkpoints",
    "discover_s1_checkpoints",
    "discover_s2_checkpoints",
    "stage_one_pairs",
    "stage_two_pairs",
]
