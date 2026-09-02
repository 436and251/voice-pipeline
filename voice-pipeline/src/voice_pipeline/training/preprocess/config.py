from __future__ import annotations

from dataclasses import dataclass
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

        root = (project_root or Path.cwd()).resolve()
        output_root = _resolve(root, output_value, "experiment.output_root")
        manifest = _resolve(root, manifest_value, "dataset.manifest")
        if not manifest.is_file():
            raise ValueError(f"dataset manifest does not exist: {manifest}")
        return cls(profile, name, output_root, manifest, root, device, precision, resume)


def _resolve(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be a path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
