from datetime import datetime
from io import StringIO
import json
from pathlib import Path

import pytest

from voice_pipeline.module_api.events import ModuleEventEmitter


def test_event_is_identical_in_stdout_and_journal(tmp_path):
    stream = StringIO()
    journal = tmp_path / "jobs" / "job-1" / "events.jsonl"
    artifact = tmp_path / "runs" / "Acane" / "candidate-a.wav"
    emitter = ModuleEventEmitter(
        "job-1",
        stream,
        journal,
        allowed_roots=(tmp_path / "runs",),
    )

    emitter.emit(
        "stage.progress",
        stage="evaluate",
        message_key="evaluation.progress",
        message_args={"candidate": "A"},
        current=1,
        total=3,
        artifacts=[{"type": "listening_audio", "path": artifact}],
    )

    stdout_line = stream.getvalue().strip()
    journal_line = journal.read_text(encoding="utf-8").strip()
    assert stdout_line == journal_line

    event = json.loads(stdout_line)
    assert event == {
        "protocol_version": 1,
        "job_id": "job-1",
        "type": "stage.progress",
        "timestamp": event["timestamp"],
        "stage": "evaluate",
        "message_key": "evaluation.progress",
        "message_args": {"candidate": "A"},
        "current": 1,
        "total": 3,
        "artifacts": [
            {"type": "listening_audio", "path": str(artifact.resolve())}
        ],
    }
    assert datetime.fromisoformat(event["timestamp"]).tzinfo is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current", float("nan")),
        ("total", float("inf")),
        ("message_args", {"loss": float("-inf")}),
    ],
)
def test_event_rejects_non_finite_numbers_without_partial_output(
    tmp_path, field, value
):
    stream = StringIO()
    journal = tmp_path / "events.jsonl"
    emitter = ModuleEventEmitter("job-1", stream, journal)

    with pytest.raises(ValueError):
        emitter.emit("stage.progress", **{field: value})

    assert stream.getvalue() == ""
    assert not journal.exists()


@pytest.mark.parametrize("path_kind", ["relative", "outside"])
def test_event_rejects_artifact_outside_allowed_roots(tmp_path, path_kind):
    stream = StringIO()
    artifact_path = (
        Path("relative.wav")
        if path_kind == "relative"
        else tmp_path.parent / "outside.wav"
    )
    emitter = ModuleEventEmitter(
        "job-1",
        stream,
        tmp_path / "jobs" / "job-1" / "events.jsonl",
        allowed_roots=(tmp_path / "runs",),
    )

    with pytest.raises(ValueError, match="artifact path"):
        emitter.emit(
            "artifact.created",
            artifacts=[{"type": "audio", "path": artifact_path}],
        )

    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": 1},
        {"message_key": ""},
        {"current": True},
        {"total": "3"},
        {"message_args": {1: "not a string key"}},
    ],
)
def test_event_rejects_invalid_field_types(tmp_path, kwargs):
    emitter = ModuleEventEmitter("job-1", StringIO(), tmp_path / "events.jsonl")

    with pytest.raises((TypeError, ValueError)):
        emitter.emit("stage_progress", **kwargs)

    assert not (tmp_path / "events.jsonl").exists()


def test_event_rejects_oversized_jsonl_line(tmp_path):
    stream = StringIO()
    journal = tmp_path / "events.jsonl"
    emitter = ModuleEventEmitter("job-1", stream, journal)

    with pytest.raises(ValueError, match="too large"):
        emitter.emit("log", message_args={"text": "x" * 70_000})

    assert stream.getvalue() == ""
    assert not journal.exists()


def test_event_is_journaled_before_stdout_write(tmp_path):
    class BrokenStream(StringIO):
        def write(self, value):
            raise OSError("stdout unavailable")

    journal = tmp_path / "events.jsonl"
    emitter = ModuleEventEmitter("job-1", BrokenStream(), journal)

    with pytest.raises(OSError, match="stdout unavailable"):
        emitter.emit("job.started")

    assert json.loads(journal.read_text(encoding="utf-8"))["type"] == "job.started"
