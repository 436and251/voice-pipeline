from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from voice_pipeline.profiles.base import ModelProfile
from voice_pipeline.profiles.registry import ProfileRegistry
from voice_pipeline.training.s1 import S1TrainConfig
from voice_pipeline.training.s2 import S2TrainConfig


_TOP_LEVEL = {"profile", "experiment", "device", "dataset", "objective", "preprocess", "s1", "s2", "evaluation"}
_S1_FIELDS = {
    "enabled", "batch_size", "gradient_accumulation", "target_optimizer_steps",
    "checkpoint_every_steps", "resume_from", "num_workers", "seed", "max_sec",
    "hz", "min_ps_ratio", "max_ps_ratio",
}
_S2_FIELDS = {
    "enabled", "batch_size", "target_steps", "checkpoint_every_steps",
    "learning_rate", "text_low_lr_rate", "freeze_quantizer", "grad_ckpt",
    "resume_from", "num_workers", "seed", "betas", "eps", "lr_decay",
    "segment_size", "c_mel", "c_kl",
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    profile: ModelProfile
    experiment_name: str
    output_root: Path
    project_root: Path
    s1: S1TrainConfig | None
    s2: S2TrainConfig | None
    s1_resume_from: Path | None
    s2_resume_from: Path | None

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path | None = None) -> "TrainingConfig":
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"invalid training YAML {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("training YAML root must be a mapping")
        _reject_unknown("top-level", payload, _TOP_LEVEL)
        _validate_shared_sections(payload)

        profile_data = _mapping(payload, "profile")
        _reject_unknown("profile", profile_data, {"name"})
        profile_name = profile_data.get("name")
        if profile_name != "v2ProPlus":
            raise ValueError(f"training supports only v2ProPlus, got {profile_name}")
        profile = ProfileRegistry.get(profile_name)

        experiment = _mapping(payload, "experiment")
        _reject_unknown("experiment", experiment, {"name", "output_root"})
        name = experiment.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("experiment.name must be a non-empty string")
        root = (project_root or Path.cwd()).resolve()
        output_root = _path(root, experiment.get("output_root"), "experiment.output_root")
        output_dir = output_root / name
        preprocess_dir = output_dir / "preprocess"

        device_data = _mapping(payload, "device")
        _reject_unknown("device", device_data, {"device", "precision"})
        device = device_data.get("device")
        precision = device_data.get("precision")
        if not isinstance(device, str) or not device:
            raise ValueError("device.device must be a non-empty string")
        if not isinstance(precision, str) or precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported precision: {precision}")

        s1, s1_resume = _parse_s1(payload.get("s1"), root, output_dir, preprocess_dir, profile, device, precision)
        s2, s2_resume = _parse_s2(payload.get("s2"), root, output_dir, preprocess_dir, profile, device, precision)
        if s1 is None and s2 is None:
            raise ValueError("at least one training stage must be enabled")
        return cls(profile, name, output_root, root, s1, s2, s1_resume, s2_resume)


def _parse_s1(value, root, output_dir, preprocess_dir, profile, device, precision):
    if value is None:
        return None, None
    section = _mapping({"s1": value}, "s1")
    _reject_unknown("s1", section, _S1_FIELDS)
    if not _enabled(section, "s1"):
        return None, None
    accumulation = _integer(section.get("gradient_accumulation", 4), "s1.gradient_accumulation")
    if accumulation != 4:
        raise ValueError("s1.gradient_accumulation must be 4")
    min_ps_ratio = _positive_number(section.get("min_ps_ratio", 3.0), "s1.min_ps_ratio")
    max_ps_ratio = _positive_number(section.get("max_ps_ratio", 25.0), "s1.max_ps_ratio")
    if min_ps_ratio > max_ps_ratio:
        raise ValueError("s1.min_ps_ratio must not exceed s1.max_ps_ratio")
    config = S1TrainConfig(
        preprocess_dir=preprocess_dir,
        output_dir=output_dir,
        base_s1_path=root / profile.s1_relative_path,
        target_optimizer_steps=_positive(section.get("target_optimizer_steps"), "s1.target_optimizer_steps"),
        checkpoint_every_steps=_positive(section.get("checkpoint_every_steps"), "s1.checkpoint_every_steps"),
        device=device,
        precision=precision,
        batch_size=_positive(section.get("batch_size", 2), "s1.batch_size"),
        num_workers=_nonnegative(section.get("num_workers", 0), "s1.num_workers"),
        seed=_integer(section.get("seed", 1234), "s1.seed"),
        gradient_accumulation=accumulation,
        max_sec=_positive(section.get("max_sec", 57), "s1.max_sec"),
        hz=_positive(section.get("hz", 25), "s1.hz"),
        min_ps_ratio=min_ps_ratio,
        max_ps_ratio=max_ps_ratio,
    )
    return config, _optional_path(root, section.get("resume_from"), "s1.resume_from")


def _parse_s2(value, root, output_dir, preprocess_dir, profile, device, precision):
    if value is None:
        return None, None
    section = _mapping({"s2": value}, "s2")
    _reject_unknown("s2", section, _S2_FIELDS)
    if not _enabled(section, "s2"):
        return None, None
    low_rate = _number(section.get("text_low_lr_rate", profile.text_low_lr_rate), "s2.text_low_lr_rate")
    if low_rate != profile.text_low_lr_rate:
        raise ValueError(f"s2.text_low_lr_rate must be {profile.text_low_lr_rate}")
    if section.get("freeze_quantizer", True) is not True:
        raise ValueError("s2.freeze_quantizer must be true for v2ProPlus")
    if section.get("grad_ckpt", False) is not False:
        raise ValueError("s2.grad_ckpt is not supported")
    betas = section.get("betas", [0.8, 0.99])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("s2.betas must contain two numbers")
    beta_values = (_number(betas[0], "s2.betas[0]"), _number(betas[1], "s2.betas[1]"))
    if any(value < 0 or value >= 1 for value in beta_values):
        raise ValueError("s2.betas values must be in [0, 1)")
    lr_decay = _number(section.get("lr_decay", 0.999875), "s2.lr_decay")
    if not 0 < lr_decay <= 1:
        raise ValueError("s2.lr_decay must be in (0, 1]")
    config = S2TrainConfig(
        preprocess_dir=preprocess_dir,
        output_dir=output_dir,
        base_s2g_path=root / profile.s2g_relative_path,
        base_s2d_path=root / profile.s2d_relative_path,
        target_optimizer_steps=_positive(section.get("target_steps"), "s2.target_steps"),
        checkpoint_every_steps=_positive(section.get("checkpoint_every_steps"), "s2.checkpoint_every_steps"),
        device=device,
        precision=precision,
        batch_size=_positive(section.get("batch_size", 2), "s2.batch_size"),
        num_workers=_nonnegative(section.get("num_workers", 0), "s2.num_workers"),
        seed=_integer(section.get("seed", 1234), "s2.seed"),
        learning_rate=_positive_number(section.get("learning_rate", 1e-4), "s2.learning_rate"),
        text_low_lr_rate=low_rate,
        betas=beta_values,
        eps=_positive_number(section.get("eps", 1e-9), "s2.eps"),
        lr_decay=lr_decay,
        segment_size=_positive(section.get("segment_size", 20480), "s2.segment_size"),
        c_mel=_nonnegative_number(section.get("c_mel", 45.0), "s2.c_mel"),
        c_kl=_nonnegative_number(section.get("c_kl", 1.0), "s2.c_kl"),
    )
    return config, _optional_path(root, section.get("resume_from"), "s2.resume_from")


def _mapping(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _reject_unknown(name: str, mapping: dict, allowed: set[str]) -> None:
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown {name} field: {', '.join(sorted(unknown))}")


def _enabled(section: dict, name: str) -> bool:
    value = section.get("enabled", True)
    if not isinstance(value, bool):
        raise ValueError(f"{name}.enabled must be boolean")
    return value


def _integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive(value, field: str) -> int:
    value = _integer(value, field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative(value, field: str) -> int:
    value = _integer(value, field)
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _positive_number(value, field: str) -> float:
    value = _number(value, field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_number(value, field: str) -> float:
    value = _number(value, field)
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _path(root: Path, value, field: str) -> Path:
    _path_value(value, field)
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _optional_path(root: Path, value, field: str) -> Path | None:
    return None if value is None else _path(root, value, field)


def _validate_shared_sections(payload: dict) -> None:
    schemas = {
        "dataset": {"manifest"},
        "objective": {"training_languages", "target_languages", "cross_language_preservation"},
        "preprocess": {"resume"},
        "evaluation": {"enabled", "reference", "suites"},
    }
    for name, allowed in schemas.items():
        if name not in payload:
            continue
        section = _mapping(payload, name)
        _reject_unknown(name, section, allowed)

    dataset = payload.get("dataset")
    if dataset is not None and "manifest" in dataset:
        _path_value(dataset["manifest"], "dataset.manifest")
    preprocess = payload.get("preprocess")
    if preprocess is not None and "resume" in preprocess and not isinstance(preprocess["resume"], bool):
        raise ValueError("preprocess.resume must be boolean")
    objective = payload.get("objective")
    if objective is not None:
        for field in ("training_languages", "target_languages"):
            if field in objective and (
                not isinstance(objective[field], list)
                or any(
                    not isinstance(language, str) or language not in {"zh", "ja", "en"}
                    for language in objective[field]
                )
            ):
                raise ValueError(f"objective.{field} must contain only zh, ja, or en")
        preservation = objective.get("cross_language_preservation")
        if preservation is not None and preservation != "strict":
            raise ValueError("objective.cross_language_preservation must be strict")
    evaluation = payload.get("evaluation")
    if evaluation is not None:
        if "enabled" in evaluation and not isinstance(evaluation["enabled"], bool):
            raise ValueError("evaluation.enabled must be boolean")
        for field, allowed in (("reference", {"audio", "text", "language"}), ("suites", {"zh", "ja", "en", "mixed"})):
            if field in evaluation:
                nested = evaluation[field]
                if not isinstance(nested, dict):
                    raise ValueError(f"evaluation.{field} must be a mapping")
                _reject_unknown(f"evaluation.{field}", nested, allowed)
        reference = evaluation.get("reference")
        if reference is not None:
            if "audio" in reference:
                _path_value(reference["audio"], "evaluation.reference.audio")
            if "text" in reference and (not isinstance(reference["text"], str) or not reference["text"].strip()):
                raise ValueError("evaluation.reference.text must be a non-empty string")
            if "language" in reference and (
                not isinstance(reference["language"], str)
                or reference["language"] not in {"zh", "ja", "en"}
            ):
                raise ValueError("evaluation.reference.language must be zh, ja, or en")
        suites = evaluation.get("suites")
        if suites is not None:
            for language, value in suites.items():
                _path_value(value, f"evaluation.suites.{language}")


def _path_value(value, field: str) -> None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")


__all__ = ["TrainingConfig"]
