from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
from pathlib import Path


class PipelineLogger:
    def __init__(
        self,
        path: Path,
        echo: bool = True,
        record_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.path = path
        self.echo = echo
        self.record_sink = record_sink

    def log(
        self,
        stage: str,
        event: str,
        *,
        mini_step: int | None = None,
        optimizer_step: int | None = None,
        metrics: Mapping[str, object] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "event": event,
            "mini_step": mini_step,
            "optimizer_step": optimizer_step,
            "metrics": dict(metrics or {}),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
        if self.record_sink is not None:
            self.record_sink(dict(record))
        if self.echo:
            print(_console_line(record), flush=True)


def _console_line(record: dict[str, object]) -> str:
    fields = [f"[{str(record['timestamp'])[11:19]}]", f"[{str(record['stage']).upper()}]", str(record["event"])]
    for name in ("mini_step", "optimizer_step"):
        if record[name] is not None:
            fields.append(f"{name}={record[name]}")
    fields.extend(f"{name}={value}" for name, value in record["metrics"].items())
    return " ".join(fields)
