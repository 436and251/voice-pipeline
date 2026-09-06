from pathlib import Path

import pytest
import torch

from voice_pipeline.evaluation.candidates import (
    CheckpointRef,
    discover_s1_checkpoints,
    discover_s2_checkpoints,
    stage_one_pairs,
    stage_two_pairs,
)


def _ref(kind: str, tmp_path: Path, step: int | None) -> CheckpointRef:
    label = "base" if step is None else str(step)
    return CheckpointRef(kind, tmp_path / f"{kind}-{label}.pt", step, label * 64 if step is None else f"{step:064x}", step is None)


def test_stage_one_uses_only_base_s1(tmp_path: Path) -> None:
    base_s1 = _ref("s1", tmp_path, None)
    s2_refs = (_ref("s2", tmp_path, None), _ref("s2", tmp_path, 100), _ref("s2", tmp_path, 200))

    pairs = stage_one_pairs(base_s1, s2_refs)

    assert [(pair.s1.step, pair.s2.step) for pair in pairs] == [(None, None), (None, 100), (None, 200)]


def test_stage_two_only_crosses_s1_with_retained_s2(tmp_path: Path) -> None:
    s1_refs = (_ref("s1", tmp_path, None), _ref("s1", tmp_path, 100), _ref("s1", tmp_path, 200))
    s2_100 = _ref("s2", tmp_path, 100)
    s2_200 = _ref("s2", tmp_path, 200)
    s2_300 = _ref("s2", tmp_path, 300)

    pairs = stage_two_pairs(s1_refs, (s2_100, s2_300))

    assert {(pair.s1.step, pair.s2.step) for pair in pairs} == {
        (None, 100), (100, 100), (200, 100),
        (None, 300), (100, 300), (200, 300),
    }
    assert all(pair.s2 != s2_200 for pair in pairs)


def test_discovery_validates_embedded_steps_and_sorts(tmp_path: Path) -> None:
    s1_dir = tmp_path / "s1"
    s2_dir = tmp_path / "s2"
    s1_dir.mkdir()
    s2_dir.mkdir()
    for step in (200, 100):
        torch.save(
            {"format_version": 1, "profile": "v2ProPlus", "optimizer_step": step, "model": {"x": torch.ones(1)}},
            s1_dir / f"step-{step:08d}.pt",
        )
        torch.save(
            {"format_version": 1, "profile": "v2ProPlus", "global_step": step, "net_g": {"x": torch.ones(1)}},
            s2_dir / f"step-{step:08d}.pt",
        )

    assert [ref.step for ref in discover_s1_checkpoints(s1_dir)] == [100, 200]
    assert [ref.step for ref in discover_s2_checkpoints(s2_dir)] == [100, 200]


def test_discovery_rejects_filename_step_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "s1"
    directory.mkdir()
    torch.save(
        {"format_version": 1, "profile": "v2ProPlus", "optimizer_step": 100, "model": {"x": torch.ones(1)}},
        directory / "step-00000200.pt",
    )

    with pytest.raises(ValueError, match="filename does not match embedded step"):
        discover_s1_checkpoints(directory)


def test_discovery_rejects_wrong_profile(tmp_path: Path) -> None:
    directory = tmp_path / "s2"
    directory.mkdir()
    torch.save(
        {"format_version": 1, "profile": "v2", "global_step": 100, "net_g": {"x": torch.ones(1)}},
        directory / "step-00000100.pt",
    )

    with pytest.raises(ValueError, match="profile"):
        discover_s2_checkpoints(directory)
