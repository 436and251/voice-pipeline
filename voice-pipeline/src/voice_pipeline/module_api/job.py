from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import re

from voice_pipeline.module_api.descriptor import build_descriptor


_FIELDS = {
    "protocol_version",
    "job_id",
    "project_name",
    "project_root",
    "output_root",
    "dataset_list",
    "dataset_sha256",
    "framework",
    "stages",
    "device",
    "precision",
    "parameters",
    "reference",
    "job_dir",
}
_STAGES = ("preprocess", "s2", "s1", "evaluate")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


@dataclass(frozen=True, slots=True)
class ModuleReference:
    audio: Path
    text: str | None
    language: str


@dataclass(frozen=True, slots=True)
class ModuleJob:
    job_id: str
    project_name: str
    project_root: Path
    output_root: Path
    dataset_list: Path
    dataset_sha256: str
    framework: str
    stages: tuple[str, ...]
    device: str
    precision: str
    parameters: dict[str, object]
    reference: ModuleReference | None
    job_dir: Path

    @classmethod
    def load(cls, path: Path) -> "ModuleJob":
        job_file = Path(path).resolve()
        try:
            payload = json.loads(job_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid job JSON {job_file}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("job JSON root must be an object")
        unknown = set(payload) - _FIELDS
        missing = _FIELDS - set(payload)
        if unknown:
            raise ValueError(f"unknown job field: {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"missing job field: {', '.join(sorted(missing))}")

        protocol_version = payload["protocol_version"]
        if type(protocol_version) is not int or protocol_version != 1:
            raise ValueError("protocol_version must be integer 1")

        job_id = _nonempty_string(payload["job_id"], "job_id")
        if _JOB_ID.fullmatch(job_id) is None:
            raise ValueError("job_id must contain only letters, numbers, dots, underscores, or hyphens")
        project_name = _nonempty_string(payload["project_name"], "project_name")
        if _PROJECT_NAME.fullmatch(project_name) is None:
            raise ValueError("project_name must contain only letters, numbers, underscores, or hyphens")
        project_root = _absolute_path(payload["project_root"], "project_root")
        output_root = _absolute_path(payload["output_root"], "output_root")
        dataset_list = _absolute_path(payload["dataset_list"], "dataset_list")
        job_dir = _absolute_path(payload["job_dir"], "job_dir")

        if not project_root.is_dir():
            raise ValueError(f"project_root does not exist: {project_root}")
        if job_dir != job_file.parent:
            raise ValueError("job_dir must equal the directory containing job.json")
        if job_dir.name != job_id:
            raise ValueError("job_id must match the job directory name")
        _require_contained(job_dir, project_root, "job_dir")
        _require_contained(output_root, project_root, "output_root")
        _require_contained(dataset_list, project_root, "dataset_list")
        if not dataset_list.is_file():
            raise ValueError(f"dataset_list does not exist: {dataset_list}")

        dataset_sha256 = payload["dataset_sha256"]
        if not isinstance(dataset_sha256, str) or _SHA256.fullmatch(dataset_sha256) is None:
            raise ValueError("dataset_sha256 must be 64 lowercase hexadecimal characters")
        if _sha256(dataset_list) != dataset_sha256:
            raise ValueError("dataset_sha256 does not match dataset_list")

        framework = _nonempty_string(payload["framework"], "framework")
        frameworks = {item["id"] for item in build_descriptor()["frameworks"]}
        if framework not in frameworks:
            raise ValueError(f"unsupported framework: {framework}")

        stages = _stages(payload["stages"])
        device = _nonempty_string(payload["device"], "device")
        precision = _nonempty_string(payload["precision"], "precision")
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported precision: {precision}")
        parameters = _parameters(payload["parameters"], framework, project_root)
        reference = _reference(
            payload["reference"],
            project_root,
            required="evaluate" in stages,
        )

        return cls(
            job_id=job_id,
            project_name=project_name,
            project_root=project_root,
            output_root=output_root,
            dataset_list=dataset_list,
            dataset_sha256=dataset_sha256,
            framework=framework,
            stages=stages,
            device=device,
            precision=precision,
            parameters=parameters,
            reference=reference,
            job_dir=job_dir,
        )


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return path.resolve()


def _require_contained(path: Path, root: Path, field: str) -> None:
    if path != root and not path.is_relative_to(root):
        raise ValueError(f"{field} must remain inside project_root")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _stages(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError("stages must be a non-empty array of stage names")
    stages = tuple(value)
    canonical = tuple(stage for stage in _STAGES if stage in stages)
    if stages != canonical:
        raise ValueError("stages must be unique and follow preprocess, s2, s1, evaluate order")
    return stages


def _parameters(value: object, framework: str, project_root: Path) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("parameters must be an object with string keys")
    framework_data = next(
        item for item in build_descriptor()["frameworks"] if item["id"] == framework
    )
    declarations = {field["key"]: field for field in framework_data["fields"]}
    unknown = set(value) - set(declarations)
    if unknown:
        raise ValueError(f"unknown parameter: {', '.join(sorted(unknown))}")
    for key, parameter in value.items():
        declaration = declarations[key]
        _validate_parameter(key, parameter, declaration, project_root)
    return dict(value)


def _reference(value: object, project_root: Path, *, required: bool) -> ModuleReference | None:
    if value is None:
        if required:
            raise ValueError("reference is required when evaluation is enabled")
        return None
    if not isinstance(value, dict):
        raise ValueError("reference must be an object")
    fields = {"audio", "text", "language"}
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise ValueError(f"unknown reference field: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing reference field: {', '.join(sorted(missing))}")
    audio = _absolute_path(value["audio"], "reference.audio")
    _require_contained(audio, project_root, "reference.audio")
    if not audio.is_file():
        raise ValueError(f"reference.audio does not exist: {audio}")
    text = value["text"]
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("reference.text must be null or a non-empty string")
    language = value["language"]
    if not isinstance(language, str) or language not in {"zh", "ja", "en"}:
        raise ValueError("reference.language must be zh, ja, or en")
    return ModuleReference(audio, text.strip() if text is not None else None, language)


def _validate_parameter(key: str, value: object, declaration: dict, root: Path) -> None:
    kind = declaration["kind"]
    try:
        allowed_types = _PARAMETER_TYPES[kind]
    except KeyError as error:
        raise ValueError(f"unsupported descriptor kind for {key}: {kind}") from error
    if type(value) not in allowed_types:
        raise ValueError(f"parameter {key} must be {kind}")
    post_validator = _PARAMETER_POST_VALIDATORS.get(kind)
    if post_validator is not None:
        post_validator(key, value, declaration, root)


def _validate_enum(key: str, value: str, declaration: dict, _root: Path) -> None:
    if value not in declaration["constraints"].get("choices", []):
        raise ValueError(f"parameter {key} must be enum")


def _validate_path(key: str, value: str, _declaration: dict, root: Path) -> None:
    if not Path(value).is_absolute():
        raise ValueError(f"parameter {key} must be path")
    _require_contained(Path(value).resolve(), root, key)


def _validate_integer(key: str, value: int, declaration: dict, _root: Path) -> None:
    _validate_numeric_constraints(key, value, declaration["constraints"])


def _validate_number(key: str, value: int | float, declaration: dict, _root: Path) -> None:
    try:
        finite = math.isfinite(float(value))
    except OverflowError as error:
        raise ValueError(f"parameter {key} must be finite") from error
    if not finite:
        raise ValueError(f"parameter {key} must be finite")
    _validate_numeric_constraints(key, value, declaration["constraints"])


def _validate_numeric_constraints(key: str, value: int | float, constraints: dict) -> None:
    for name, (compare, wording) in _NUMERIC_CONSTRAINTS.items():
        if name in constraints and not compare(value, constraints[name]):
            raise ValueError(f"parameter {key} must be {wording} {constraints[name]}")


_PARAMETER_TYPES = {
    "boolean": {bool},
    "integer": {int},
    "number": {int, float},
    "string": {str},
    "enum": {str},
    "path": {str},
}
_PARAMETER_POST_VALIDATORS = {
    "integer": _validate_integer,
    "number": _validate_number,
    "enum": _validate_enum,
    "path": _validate_path,
}
_NUMERIC_CONSTRAINTS = {
    "minimum": (operator.ge, "at least"),
    "maximum": (operator.le, "at most"),
    "exclusive_minimum": (operator.gt, "greater than"),
    "exclusive_maximum": (operator.lt, "less than"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ModuleJob", "ModuleReference"]
