from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TextIO

MAX_EVENT_BYTES = 64 * 1024


class ModuleEventEmitter:
    """Write one durable JSONL event to the job journal and host stream."""

    def __init__(
        self,
        job_id: str,
        stream: TextIO,
        journal: Path,
        *,
        allowed_roots: Sequence[Path] | None = None,
    ) -> None:
        _require_text("job_id", job_id)
        self.job_id = job_id
        self.stream = stream
        self.journal = journal
        roots = allowed_roots if allowed_roots is not None else (journal.parent,)
        self.allowed_roots = tuple(Path(root).resolve() for root in roots)
        if not self.allowed_roots:
            raise ValueError("allowed_roots must not be empty")

    def emit(
        self,
        type: str,
        *,
        stage: str | None = None,
        message_key: str | None = None,
        message_args: Mapping[str, object] | None = None,
        current: int | float | None = None,
        total: int | float | None = None,
        artifacts: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        _require_text("event type", type)
        if stage is not None:
            _require_text("stage", stage)
        if message_key is not None:
            _require_text("message_key", message_key)
        if message_args is not None:
            if not isinstance(message_args, Mapping):
                raise TypeError("message_args must be an object")
            if not all(isinstance(key, str) for key in message_args):
                raise TypeError("message_args keys must be strings")
        _validate_progress("current", current)
        _validate_progress("total", total)
        event: dict[str, object] = {
            "protocol_version": 1,
            "job_id": self.job_id,
            "type": type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        optional = {
            "stage": stage,
            "message_key": message_key,
            "message_args": dict(message_args) if message_args is not None else None,
            "current": current,
            "total": total,
            "artifacts": self._normalize_artifacts(artifacts),
        }
        event.update({name: value for name, value in optional.items() if value is not None})
        line = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len((line + "\n").encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError(f"event line is too large (maximum {MAX_EVENT_BYTES} bytes)")

        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8", newline="\n") as journal_stream:
            journal_stream.write(line + "\n")
            journal_stream.flush()
        self.stream.write(line + "\n")
        self.stream.flush()

    def _normalize_artifacts(
        self, artifacts: Sequence[Mapping[str, object]] | None
    ) -> list[dict[str, object]] | None:
        if artifacts is None:
            return None
        normalized = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise TypeError("artifact must be an object")
            if not isinstance(artifact.get("type"), str) or not artifact["type"]:
                raise ValueError("artifact type must be a non-empty string")
            raw_path = artifact.get("path")
            if not isinstance(raw_path, (str, Path)):
                raise ValueError("artifact path must be a string or Path")
            path = Path(raw_path)
            if not path.is_absolute():
                raise ValueError("artifact path must be absolute")
            resolved = path.resolve()
            if not any(resolved.is_relative_to(root) for root in self.allowed_roots):
                raise ValueError("artifact path is outside allowed roots")
            normalized.append({**artifact, "path": str(resolved)})
        return normalized


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_progress(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
