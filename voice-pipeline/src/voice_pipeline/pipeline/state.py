from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from voice_pipeline.pipeline.config import PipelineSpec


_STATUSES = {"pending", "running", "completed", "failed"}


@dataclass(slots=True)
class PipelineState:
    path: Path
    _identity: dict[str, object]
    _stages: dict[str, str]
    failure: dict[str, str] | None = None

    @classmethod
    def load_or_create(cls, path: Path, spec: PipelineSpec) -> "PipelineState":
        path = Path(path).resolve()
        identity: dict[str, object] = {
            "pipeline": spec.path.as_posix(),
            "config": spec.training_config.as_posix(),
            "stages": list(spec.stages),
        }
        if not path.exists():
            state = cls(
                path=path,
                _identity=identity,
                _stages={stage: "pending" for stage in spec.stages},
            )
            state._persist()
            return state

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid pipeline state {path}: {error}") from error
        cls._validate_payload(payload, identity)
        return cls(
            path=path,
            _identity=identity,
            _stages=dict(payload["stages"]),
            failure=None if payload["failure"] is None else dict(payload["failure"]),
        )

    @staticmethod
    def _validate_payload(payload: object, identity: dict[str, object]) -> None:
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "identity",
            "stages",
            "failure",
        }:
            raise ValueError("invalid pipeline state fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("invalid pipeline state schema_version")
        if payload["identity"] != identity:
            raise ValueError("pipeline state identity does not match pipeline specification")

        stages = payload["stages"]
        expected_stages = identity["stages"]
        if (
            not isinstance(stages, dict)
            or list(stages) != expected_stages
            or any(type(status) is not str or status not in _STATUSES for status in stages.values())
        ):
            raise ValueError("invalid pipeline state stages")

        failure = payload["failure"]
        if failure is None:
            return
        if (
            not isinstance(failure, dict)
            or set(failure) != {"stage", "type", "message"}
            or any(type(value) is not str for value in failure.values())
            or failure["stage"] not in stages
            or stages[failure["stage"]] != "failed"
        ):
            raise ValueError("invalid pipeline state failure")

    def status(self, stage: str) -> str:
        self._require_stage(stage)
        return self._stages[stage]

    def start(self, stage: str) -> None:
        self._require_stage(stage)
        self._stages[stage] = "running"
        self.failure = None
        self._persist()

    def complete(self, stage: str) -> None:
        self._require_stage(stage)
        self._stages[stage] = "completed"
        self.failure = None
        self._persist()

    def fail(self, stage: str, error: Exception) -> None:
        self._require_stage(stage)
        self._stages[stage] = "failed"
        self.failure = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        self._persist()

    def _require_stage(self, stage: str) -> None:
        if stage not in self._stages:
            raise ValueError(f"pipeline stage is not declared: {stage}")

    def _persist(self) -> None:
        payload = {
            "schema_version": 1,
            "identity": self._identity,
            "stages": self._stages,
            "failure": self.failure,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
