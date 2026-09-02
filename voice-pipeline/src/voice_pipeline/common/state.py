from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import uuid


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


_ALLOWED = {
    StageStatus.PENDING: {StageStatus.RUNNING},
    StageStatus.RUNNING: {StageStatus.COMPLETED, StageStatus.FAILED},
    StageStatus.COMPLETED: {StageStatus.INVALIDATED},
    StageStatus.FAILED: {StageStatus.RUNNING, StageStatus.INVALIDATED},
    StageStatus.INVALIDATED: {StageStatus.RUNNING},
}


@dataclass(slots=True)
class StageState:
    name: str
    status: StageStatus = StageStatus.PENDING
    signature: str | None = None
    outputs: list[str] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    warning_count: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.outputs is None:
            self.outputs = []


class RunState:
    def __init__(self, path: Path):
        self.path = path
        self._stages: dict[str, StageState] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._stages = {
                    name: StageState(
                        name=name,
                        status=StageStatus(value["status"]),
                        signature=value.get("signature"),
                        outputs=list(value.get("outputs", [])),
                        started_at=value.get("started_at"),
                        finished_at=value.get("finished_at"),
                        warning_count=int(value.get("warning_count", 0)),
                        error=value.get("error"),
                    )
                    for name, value in data.get("stages", {}).items()
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid run state file: {path}: {error}") from error

    def get(self, name: str) -> StageState:
        return self._stages.setdefault(name, StageState(name))

    def transition(self, name: str, new_status: StageStatus) -> None:
        stage = self.get(name)
        if new_status not in _ALLOWED[stage.status]:
            raise ValueError(f"illegal stage transition: {stage.status.value} -> {new_status.value}")
        stage.status = new_status
        self._save()

    def start(self, name: str, *, signature: str) -> None:
        self.transition(name, StageStatus.RUNNING)
        stage = self.get(name)
        stage.signature = signature
        stage.outputs = []
        stage.started_at = _now()
        stage.finished_at = None
        stage.warning_count = 0
        stage.error = None
        self._save()

    def complete(self, name: str, *, outputs: list[str], warning_count: int = 0) -> None:
        self.transition(name, StageStatus.COMPLETED)
        stage = self.get(name)
        stage.outputs = list(outputs)
        stage.finished_at = _now()
        stage.warning_count = warning_count
        stage.error = None
        self._save()

    def fail(self, name: str, error: str) -> None:
        self.transition(name, StageStatus.FAILED)
        stage = self.get(name)
        stage.finished_at = _now()
        stage.error = error
        self._save()

    def invalidate(self, name: str) -> None:
        self.transition(name, StageStatus.INVALIDATED)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stages": {
                name: {
                    "status": stage.status.value,
                    "signature": stage.signature,
                    "outputs": stage.outputs,
                    "started_at": stage.started_at,
                    "finished_at": stage.finished_at,
                    "warning_count": stage.warning_count,
                    "error": stage.error,
                }
                for name, stage in self._stages.items()
            }
        }
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
