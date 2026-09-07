from dataclasses import dataclass
from pathlib import Path

import yaml


CANONICAL_STAGES = ("preprocess", "s2", "s1", "evaluate")


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    path: Path
    training_config: Path
    stages: tuple[str, ...]

    @classmethod
    def load(cls, path: Path, project_root: Path) -> "PipelineSpec":
        path = Path(path).resolve()
        root = Path(project_root).resolve()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"invalid pipeline YAML {path}: {error}") from error

        if not isinstance(payload, dict):
            raise ValueError("pipeline YAML root must be a mapping")
        unknown = set(payload) - {"schema_version", "config", "stages"}
        if unknown:
            raise ValueError(f"unknown pipeline field: {', '.join(sorted(unknown))}")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("pipeline.schema_version must be 1")

        raw_config = payload.get("config")
        if not isinstance(raw_config, str) or not raw_config.strip():
            raise ValueError("pipeline.config must be a non-empty path")
        training_config = (root / raw_config).resolve()
        try:
            training_config.relative_to(root)
        except ValueError as error:
            raise ValueError("pipeline.config must remain inside project_root") from error
        if not training_config.is_file():
            raise FileNotFoundError(training_config)

        stages = payload.get("stages")
        if (
            not isinstance(stages, list)
            or not stages
            or any(not isinstance(stage, str) for stage in stages)
        ):
            raise ValueError("pipeline.stages must be a non-empty list of strings")
        if len(set(stages)) != len(stages):
            raise ValueError("pipeline.stages must not contain duplicates")
        unknown_stages = set(stages) - set(CANONICAL_STAGES)
        if unknown_stages:
            raise ValueError(
                f"unknown pipeline stage: {', '.join(sorted(unknown_stages))}"
            )
        indexes = [CANONICAL_STAGES.index(stage) for stage in stages]
        if indexes != sorted(indexes):
            raise ValueError("pipeline.stages must follow canonical order")

        return cls(path, training_config, tuple(stages))
