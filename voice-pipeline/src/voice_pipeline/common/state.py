from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


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


class RunState:
    def __init__(self, path: Path):
        self.path = path
        self._stages: dict[str, StageState] = {}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._stages = {k: StageState(k, StageStatus(v["status"])) for k, v in data.get("stages", {}).items()}

    def get(self, name: str) -> StageState:
        return self._stages.setdefault(name, StageState(name))

    def transition(self, name: str, new_status: StageStatus) -> None:
        stage = self.get(name)
        if new_status not in _ALLOWED[stage.status]:
            raise ValueError(f"illegal stage transition: {stage.status.value} -> {new_status.value}")
        stage.status = new_status
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stages": {name: {"status": stage.status.value} for name, stage in self._stages.items()}}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
