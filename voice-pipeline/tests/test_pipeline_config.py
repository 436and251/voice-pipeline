import json
import os
from pathlib import Path

import pytest

from voice_pipeline.pipeline.config import PipelineSpec
from voice_pipeline.pipeline.state import PipelineState


def _write_pipeline(tmp_path: Path, stages: str) -> Path:
    (tmp_path / "train.yaml").write_text("training", encoding="utf-8")
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "schema_version: 1\nconfig: train.yaml\nstages:\n" + stages,
        encoding="utf-8",
    )
    return path


def test_pipeline_spec_resolves_config_and_preserves_canonical_order(
    tmp_path: Path,
) -> None:
    spec = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n  - evaluate\n"),
        tmp_path,
    )

    assert spec.path == (tmp_path / "pipeline.yaml").resolve()
    assert spec.training_config == (tmp_path / "train.yaml").resolve()
    assert spec.stages == ("preprocess", "s2", "evaluate")


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        ("", "non-empty"),
        ("  - preprocess\n  - preprocess\n", "duplicate"),
        ("  - preprocess\n  - export\n", "unknown"),
        ("  - s1\n  - s2\n", "canonical order"),
    ],
)
def test_pipeline_spec_rejects_invalid_stages(
    tmp_path: Path, stages: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PipelineSpec.load(_write_pipeline(tmp_path, stages), tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("- preprocess\n", "root must be a mapping"),
        ("schema_version: 2\nconfig: train.yaml\nstages: [s1]\n", "must be 1"),
        ("schema_version: true\nconfig: train.yaml\nstages: [s1]\n", "must be 1"),
        (
            "schema_version: 1\nconfig: train.yaml\nstages: [s1]\nextra: true\n",
            "unknown pipeline field",
        ),
    ],
)
def test_pipeline_spec_rejects_invalid_root_fields(
    tmp_path: Path, content: str, message: str
) -> None:
    (tmp_path / "train.yaml").write_text("training", encoding="utf-8")
    path = tmp_path / "pipeline.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PipelineSpec.load(path, tmp_path)


@pytest.mark.parametrize("config", ["", "missing.yaml"])
def test_pipeline_spec_rejects_missing_training_config(
    tmp_path: Path, config: str
) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        f"schema_version: 1\nconfig: {config!r}\nstages: [preprocess]\n",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        PipelineSpec.load(path, tmp_path)


def test_pipeline_spec_rejects_training_config_directory(tmp_path: Path) -> None:
    (tmp_path / "config-dir").mkdir()
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "schema_version: 1\nconfig: config-dir\nstages: [preprocess]\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        PipelineSpec.load(path, tmp_path)


def test_pipeline_spec_rejects_config_outside_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (tmp_path / "outside.yaml").write_text("training", encoding="utf-8")
    path = project_root / "pipeline.yaml"
    path.write_text(
        "schema_version: 1\nconfig: ../outside.yaml\nstages: [preprocess]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inside project_root"):
        PipelineSpec.load(path, project_root)


def test_pipeline_state_persists_transitions_and_failure(tmp_path: Path) -> None:
    spec = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n"), tmp_path
    )
    path = tmp_path / "run" / "pipeline-state.json"
    state = PipelineState.load_or_create(path, spec)

    state.start("preprocess")
    state.complete("preprocess")
    state.start("s2")
    state.fail("s2", RuntimeError("CUDA overflow"))

    loaded = PipelineState.load_or_create(path, spec)
    assert loaded.status("preprocess") == "completed"
    assert loaded.status("s2") == "failed"
    assert loaded.failure == {
        "stage": "s2",
        "type": "RuntimeError",
        "message": "CUDA overflow",
    }


def test_pipeline_state_rejects_identity_mismatch(tmp_path: Path) -> None:
    first = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n"), tmp_path
    )
    state_path = tmp_path / "run" / "pipeline-state.json"
    PipelineState.load_or_create(state_path, first)
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(
        "schema_version: 1\nconfig: train.yaml\nstages: [preprocess, s2, s1]\n",
        encoding="utf-8",
    )
    changed = PipelineSpec.load(changed_path, tmp_path)

    with pytest.raises(ValueError, match="identity"):
        PipelineState.load_or_create(state_path, changed)


def test_start_clears_previous_failure_without_corrupting_state(tmp_path: Path) -> None:
    spec = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n"), tmp_path
    )
    state_path = tmp_path / "run" / "pipeline-state.json"
    state = PipelineState.load_or_create(state_path, spec)
    state.fail("preprocess", RuntimeError("retry elsewhere"))

    state.start("s2")

    loaded = PipelineState.load_or_create(state_path, spec)
    assert loaded.failure is None
    assert loaded.status("preprocess") == "failed"
    assert loaded.status("s2") == "running"


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(extra=True),
        lambda payload: payload.update(stages={"preprocess": "unknown"}),
        lambda payload: payload.update(failure={"stage": "preprocess"}),
    ],
)
def test_pipeline_state_rejects_invalid_persisted_schema(tmp_path: Path, change) -> None:
    spec = PipelineSpec.load(_write_pipeline(tmp_path, "  - preprocess\n"), tmp_path)
    state_path = tmp_path / "run" / "pipeline-state.json"
    PipelineState.load_or_create(state_path, spec)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    change(payload)
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="state"):
        PipelineState.load_or_create(state_path, spec)


def test_pipeline_state_rejects_undeclared_stage(tmp_path: Path) -> None:
    spec = PipelineSpec.load(_write_pipeline(tmp_path, "  - preprocess\n"), tmp_path)
    state = PipelineState.load_or_create(tmp_path / "run" / "state.json", spec)

    with pytest.raises(ValueError, match="not declared"):
        state.status("s2")
    with pytest.raises(ValueError, match="not declared"):
        state.start("s2")
    with pytest.raises(ValueError, match="not declared"):
        state.complete("s2")
    with pytest.raises(ValueError, match="not declared"):
        state.fail("s2", RuntimeError("unused"))


def test_pipeline_state_publishes_each_mutation_with_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = PipelineSpec.load(_write_pipeline(tmp_path, "  - preprocess\n"), tmp_path)
    state_path = tmp_path / "run" / "pipeline-state.json"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.is_file()
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr("voice_pipeline.pipeline.state.os.replace", recording_replace)
    state = PipelineState.load_or_create(state_path, spec)
    state.start("preprocess")
    state.complete("preprocess")

    assert len(replacements) == 3
    for source, destination in replacements:
        assert source.parent == state_path.parent
        assert source.name.startswith(f".{state_path.name}.")
        assert source.suffix == ".tmp"
        assert destination == state_path
