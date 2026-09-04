from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from voice_pipeline.profiles.base import ModelProfile
from voice_pipeline.profiles.registry import ProfileRegistry


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    profile: ModelProfile
    experiment_name: str
    output_root: Path
    manifest: Path
    project_root: Path
    device: str
    precision: str
    resume: bool
    validate_s1: bool
    validate_s2: bool
    s1_max_sec: int
    s1_hz: int
    s1_min_ps_ratio: float
    s1_max_ps_ratio: float

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path | None = None) -> "PreprocessConfig":
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"invalid preprocessing YAML {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"invalid preprocessing YAML {path}: root must be a mapping")

        try:
            profile_value = payload["profile"]
            profile_name = profile_value["name"] if isinstance(profile_value, dict) else profile_value
            if profile_name != "v2ProPlus":
                raise ValueError(f"preprocessing supports only v2ProPlus, got {profile_name}")
            profile = ProfileRegistry.get(profile_name)
            experiment = payload["experiment"]
            device_config = payload["device"]
            dataset = payload["dataset"]
            preprocess = payload["preprocess"]
            name = experiment["name"]
            output_value = experiment["output_root"]
            manifest_value = dataset["manifest"]
            device = device_config["device"]
            precision = device_config["precision"]
            resume = preprocess["resume"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid preprocessing YAML {path}: missing or malformed field {error}") from error

        if not isinstance(name, str) or not name.strip():
            raise ValueError("experiment.name must be a non-empty string")
        if not isinstance(device, str) or not device:
            raise ValueError("device.device must be a non-empty string")
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported precision: {precision}")
        if not isinstance(resume, bool):
            raise ValueError("preprocess.resume must be boolean")
        s1 = _optional_mapping(payload.get("s1"), "s1")
        s2 = _optional_mapping(payload.get("s2"), "s2")
        validate_s1 = _enabled(s1, "s1")
        validate_s2 = _enabled(s2, "s2")
        s1_max_sec = _positive_int(s1.get("max_sec", 57), "s1.max_sec")
        s1_hz = _positive_int(s1.get("hz", 25), "s1.hz")
        s1_min_ps_ratio = _positive_number(s1.get("min_ps_ratio", 3.0), "s1.min_ps_ratio")
        s1_max_ps_ratio = _positive_number(s1.get("max_ps_ratio", 25.0), "s1.max_ps_ratio")
        if s1_min_ps_ratio > s1_max_ps_ratio:
            raise ValueError("s1.min_ps_ratio must not exceed s1.max_ps_ratio")

        root = (project_root or Path.cwd()).resolve()
        output_root = _resolve(root, output_value, "experiment.output_root")
        manifest = _resolve(root, manifest_value, "dataset.manifest")
        if not manifest.is_file():
            raise ValueError(f"dataset manifest does not exist: {manifest}")
        return cls(
            profile,
            name,
            output_root,
            manifest,
            root,
            device,
            precision,
            resume,
            validate_s1,
            validate_s2,
            s1_max_sec,
            s1_hz,
            s1_min_ps_ratio,
            s1_max_ps_ratio,
        )


def _resolve(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be a path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _optional_mapping(value: object, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _enabled(section: dict, field: str) -> bool:
    value = section.get("enabled", True)
    if not isinstance(value, bool):
        raise ValueError(f"{field}.enabled must be boolean")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a positive finite number") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return result
