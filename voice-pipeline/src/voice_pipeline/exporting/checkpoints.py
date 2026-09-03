from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import uuid

import torch

from voice_pipeline.core.gpt_sovits.compatibility.checkpoints import load_sovits, save_sovits


def export_s1_checkpoint(source: Path, base: Path, destination: Path) -> dict:
    source, base, destination = _paths(source, base, destination)
    base_payload = _official_envelope(_torch_load(base), "S1 base")
    if source == base:
        weights = base_payload["weight"]
        source_kind = "official_base"
        step = None
    else:
        payload = _torch_load(source)
        _training_envelope(payload, "S1", "optimizer_step")
        weights = _weights(payload.get("model"), "S1 model")
        source_kind = "training_checkpoint"
        step = payload["optimizer_step"]
    normalized = _normalize_s1(weights)
    converted_weights = {key: _half(value, f"S1 weight {key}") for key, value in normalized.items()}
    expected_weights = _normalize_s1(base_payload["weight"])
    _validate_compatible("S1", converted_weights, expected_weights)
    converted = {
        "weight": converted_weights,
        "config": base_payload["config"],
        "info": "voice-pipeline v2ProPlus inference export",
    }
    _atomic_torch_save(converted, destination)
    return _metadata(source, destination, source_kind, step)


def export_s2_checkpoint(source: Path, base: Path, destination: Path) -> dict:
    source, base, destination = _paths(source, base, destination)
    base_payload = _official_envelope(load_sovits(base), "S2 generator base")
    if source == base:
        weights = base_payload["weight"]
        source_kind = "official_base"
        step = None
    else:
        payload = _torch_load(source)
        _training_envelope(payload, "S2", "global_step")
        if "net_g" not in payload:
            raise ValueError("S2 training checkpoint is missing generator weights")
        weights = _weights(payload["net_g"], "S2 generator")
        source_kind = "training_checkpoint"
        step = payload["global_step"]
    converted_weights = {
        key: _half(value, f"S2 weight {key}")
        for key, value in weights.items()
        if "enc_q" not in key
    }
    if not converted_weights:
        raise ValueError("S2 generator has no inference weights")
    expected_weights = {key: value for key, value in base_payload["weight"].items() if "enc_q" not in key}
    _validate_compatible("S2", converted_weights, expected_weights)
    converted = {
        "weight": converted_weights,
        "config": base_payload["config"],
        "info": "voice-pipeline v2ProPlus inference export",
    }
    _atomic_sovits_save(converted, destination)
    return _metadata(source, destination, source_kind, step)


def _paths(source: Path, base: Path, destination: Path) -> tuple[Path, Path, Path]:
    source = Path(source).resolve()
    base = Path(base).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise ValueError(f"source checkpoint does not exist: {source.name}")
    if not base.is_file():
        raise ValueError(f"base checkpoint does not exist: {base.name}")
    if destination in {source, base}:
        raise ValueError("source, base, and destination paths must differ")
    return source, base, destination


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"invalid checkpoint {path.name}: {error}") from error


def _official_envelope(payload: object, name: str) -> Mapping:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid {name}: missing config")
    _weights(payload.get("weight"), f"{name} weight")
    return payload


def _training_envelope(payload: object, name: str, step_field: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid {name} training checkpoint")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported {name} checkpoint format_version")
    if payload.get("profile") != "v2ProPlus":
        raise ValueError(f"{name} checkpoint profile must be v2ProPlus")
    step = payload.get(step_field)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid {name} optimizer step")


def _weights(value: object, name: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"invalid {name} weights")
    if any(not isinstance(key, str) or not isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise ValueError(f"invalid {name} weights")
    return value


def _half(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.is_complex():
        raise ValueError(f"{name} cannot be complex")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if tensor.is_floating_point():
        tensor = tensor.half()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must remain finite in FP16")
    return tensor


def _s1_key(key: str) -> str:
    return key if key.startswith("model.") else f"model.{key}"


def _normalize_s1(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in weights.items():
        key = _s1_key(key)
        if key in normalized:
            raise ValueError(f"duplicate S1 weight key after normalization: {key}")
        normalized[key] = tensor
    return normalized


def _validate_compatible(name: str, actual: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]) -> None:
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"{name} inference weight keys do not match pinned base; missing={missing}, extra={extra}")
    for key, tensor in actual.items():
        expected_tensor = expected[key]
        if tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"{name} inference weight shape does not match pinned base for {key}: "
                f"{tuple(tensor.shape)} != {tuple(expected_tensor.shape)}"
            )
        if expected_tensor.is_floating_point():
            compatible_dtype = tensor.is_floating_point()
        else:
            compatible_dtype = tensor.dtype == expected_tensor.dtype
        if not compatible_dtype:
            raise ValueError(
                f"{name} inference weight dtype does not match pinned base for {key}: "
                f"{tensor.dtype} is incompatible with {expected_tensor.dtype}"
            )


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    _atomic_save(payload, destination, torch.save)


def _atomic_sovits_save(payload: dict, destination: Path) -> None:
    _atomic_save(payload, destination, lambda value, path: save_sovits(path, value))


def _atomic_save(payload: dict, destination: Path, saver) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        saver(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(source: Path, destination: Path, source_kind: str, step: int | None) -> dict:
    return {
        "source_sha256": _sha256(source),
        "exported_sha256": _sha256(destination),
        "source_kind": source_kind,
        "optimizer_step": step,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["export_s1_checkpoint", "export_s2_checkpoint"]
