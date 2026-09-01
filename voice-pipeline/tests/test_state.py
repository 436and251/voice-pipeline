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
