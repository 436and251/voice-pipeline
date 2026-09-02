from pathlib import Path
import pytest

from voice_pipeline.common.state import RunState, StageStatus


def test_stage_state_transitions_and_persistence(tmp_path: Path):
    path = tmp_path / "state.json"
    state = RunState(path)
    state.transition("sv", StageStatus.RUNNING)
    state.transition("sv", StageStatus.COMPLETED)

    loaded = RunState(path)
    assert loaded.get("sv").status is StageStatus.COMPLETED


def test_completed_stage_can_be_invalidated(tmp_path: Path):
    state = RunState(tmp_path / "state.json")
    state.transition("text", StageStatus.RUNNING)
    state.transition("text", StageStatus.COMPLETED)
    state.transition("text", StageStatus.INVALIDATED)
    assert state.get("text").status is StageStatus.INVALIDATED


def test_illegal_transition_is_rejected(tmp_path: Path):
    state = RunState(tmp_path / "state.json")
    with pytest.raises(ValueError):
        state.transition("s1", StageStatus.COMPLETED)


def test_stage_metadata_round_trips(tmp_path: Path):
    path = tmp_path / "state.json"
    state = RunState(path)
    state.start("text", signature="abc")
    state.complete("text", outputs=["preprocess/text/index.jsonl"], warning_count=2)
    loaded = RunState(path).get("text")
    assert loaded.status is StageStatus.COMPLETED
    assert loaded.signature == "abc"
    assert loaded.outputs == ["preprocess/text/index.jsonl"]
    assert loaded.started_at is not None
    assert loaded.finished_at is not None
    assert loaded.warning_count == 2
    assert loaded.error is None


def test_failed_stage_records_error_and_can_restart(tmp_path: Path):
    state = RunState(tmp_path / "state.json")
    state.start("hubert", signature="first")
    state.fail("hubert", "CUDA out of memory")
    assert state.get("hubert").error == "CUDA out of memory"
    state.start("hubert", signature="second")
    assert state.get("hubert").status is StageStatus.RUNNING
    assert state.get("hubert").signature == "second"
    assert state.get("hubert").error is None


def test_corrupt_state_names_the_file(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="state.json"):
        RunState(path)
