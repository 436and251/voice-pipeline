import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_pipeline.pipeline import orchestrator
from voice_pipeline.pipeline.orchestrator import PipelineStageError, run_pipeline


def _pipeline_fixture(tmp_path: Path, stages: list[str]) -> Path:
    (tmp_path / "train.yaml").write_text("training", encoding="utf-8")
    pipeline = tmp_path / "pipeline.yaml"
    lines = "\n".join(f"  - {stage}" for stage in stages)
    pipeline.write_text(
        f"schema_version: 1\nconfig: train.yaml\nstages:\n{lines}\n",
        encoding="utf-8",
    )
    return pipeline


def test_run_pipeline_executes_declared_stages_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "s2", "s1"])
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator, "_run_dir", lambda config, root: tmp_path / "runs" / "speaker"
    )

    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: calls.append(stage),
    )

    assert calls == ["preprocess", "s2", "s1"]
    assert outcome.executed == ("preprocess", "s2", "s1")
    assert outcome.skipped == ()
    assert outcome.state_path == (
        tmp_path / "runs" / "speaker" / "pipeline-state.json"
    ).resolve()
    assert outcome.cleaned is False


def test_failure_stops_later_stages_and_rerun_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "s2", "s1"])
    run_dir = tmp_path / "runs" / "speaker"
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: run_dir)
    first_calls: list[str] = []

    def fail_s2(stage: str, config: Path, root: Path) -> None:
        first_calls.append(stage)
        if stage == "s2":
            raise RuntimeError("failed S2")

    with pytest.raises(PipelineStageError, match="s2.*failed S2"):
        run_pipeline(pipeline, tmp_path, execute_stage=fail_s2)
    assert first_calls == ["preprocess", "s2"]

    retry_calls: list[str] = []
    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: retry_calls.append(stage),
    )

    assert retry_calls == ["s2", "s1"]
    assert outcome.executed == ("s2", "s1")
    assert outcome.skipped == ("preprocess",)


def test_running_stage_is_retried_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "s2"])
    run_dir = tmp_path / "runs" / "speaker"
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: run_dir)
    state_path = run_dir / "pipeline-state.json"

    def interrupt(stage: str, config: Path, root: Path) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(pipeline, tmp_path, execute_stage=interrupt)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["stages"]["preprocess"] == "running"

    calls: list[str] = []
    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: calls.append(stage),
    )

    assert calls == ["preprocess", "s2"]
    assert outcome.executed == ("preprocess", "s2")


def test_execute_preprocess_runs_full_pipeline_and_publishes_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_pipeline.training import manifest as manifest_module
    from voice_pipeline.training.preprocess import config as config_module
    from voice_pipeline.training.preprocess import factory, indexes

    records = (SimpleNamespace(sample_id="one"), SimpleNamespace(sample_id="two"))
    issues = (SimpleNamespace(message="bad"),)
    manifest = SimpleNamespace(records=records, issues=issues)
    config = SimpleNamespace(manifest=tmp_path / "data.list")
    summary = SimpleNamespace(valid_sample_ids=("two",))
    pipeline = SimpleNamespace(
        context=SimpleNamespace(preprocess_dir=tmp_path / "run" / "preprocess")
    )
    calls: list[tuple] = []
    pipeline.run = lambda loaded_records, loaded_issues: calls.append(
        ("run", loaded_records, loaded_issues)
    ) or summary
    monkeypatch.setattr(
        config_module.PreprocessConfig,
        "from_yaml",
        classmethod(
            lambda cls, path, project_root=None: calls.append(
                ("config", path, project_root)
            )
            or config
        ),
    )
    monkeypatch.setattr(
        manifest_module,
        "read_manifest_records",
        lambda path: calls.append(("manifest", path)) or manifest,
    )
    monkeypatch.setattr(
        factory,
        "build_preprocess_pipeline",
        lambda loaded: calls.append(("build", loaded)) or pipeline,
    )
    monkeypatch.setattr(
        indexes,
        "publish_training_indexes",
        lambda directory, loaded_records, valid: calls.append(
            ("publish", directory, loaded_records, valid)
        ),
    )

    orchestrator.execute_stage("preprocess", tmp_path / "train.yaml", tmp_path)

    assert calls == [
        ("config", tmp_path / "train.yaml", tmp_path),
        ("manifest", config.manifest),
        ("build", config),
        ("run", records, issues),
        ("publish", pipeline.context.preprocess_dir, records, {"two"}),
    ]


@pytest.mark.parametrize(
    ("stage", "config_field", "resume_field"),
    [("s2", "s2", "s2_resume_from"), ("s1", "s1", "s1_resume_from")],
)
def test_execute_training_stage_uses_only_matching_trainer_and_explicit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    config_field: str,
    resume_field: str,
) -> None:
    from voice_pipeline.training import config as config_module
    from voice_pipeline.training import s1 as s1_module
    from voice_pipeline.training import s2 as s2_module

    stage_config = object()
    resume = tmp_path / "resume.pt"
    training = SimpleNamespace(
        s1=stage_config if stage == "s1" else None,
        s2=stage_config if stage == "s2" else None,
        s1_resume_from=resume if stage == "s1" else None,
        s2_resume_from=resume if stage == "s2" else None,
    )
    calls: list[tuple] = []
    monkeypatch.setattr(
        config_module.TrainingConfig,
        "from_yaml",
        classmethod(
            lambda cls, path, project_root=None: calls.append(
                ("config", path, project_root)
            )
            or training
        ),
    )

    class Trainer:
        @classmethod
        def from_pretrained(cls, loaded, *, resume_from=None):
            calls.append(("build", loaded, resume_from))
            return cls()

        def train(self) -> None:
            calls.append(("train",))

    class WrongTrainer:
        @classmethod
        def from_pretrained(cls, loaded, *, resume_from=None):
            raise AssertionError("the unrelated trainer must not be constructed")

    monkeypatch.setattr(s1_module, "S1Trainer", Trainer if stage == "s1" else WrongTrainer)
    monkeypatch.setattr(s2_module, "S2Trainer", Trainer if stage == "s2" else WrongTrainer)

    orchestrator.execute_stage(stage, tmp_path / "train.yaml", tmp_path)

    assert getattr(training, config_field) is stage_config
    assert getattr(training, resume_field) == resume
    assert calls == [
        ("config", tmp_path / "train.yaml", tmp_path),
        ("build", stage_config, resume),
        ("train",),
    ]


@pytest.mark.parametrize("stage", ["s1", "s2"])
def test_execute_training_stage_rejects_disabled_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from voice_pipeline.training import config as config_module

    training = SimpleNamespace(
        s1=None,
        s2=None,
        s1_resume_from=None,
        s2_resume_from=None,
    )
    monkeypatch.setattr(
        config_module.TrainingConfig,
        "from_yaml",
        classmethod(lambda cls, path, project_root=None: training),
    )

    with pytest.raises(ValueError, match=f"{stage.upper()} training is disabled"):
        orchestrator.execute_stage(stage, tmp_path / "train.yaml", tmp_path)


def test_execute_evaluate_calls_configured_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_pipeline.evaluation import pipeline as evaluation_module
    from voice_pipeline.training import config as config_module

    evaluation = object()
    training = SimpleNamespace(evaluation=evaluation)
    calls: list[object] = []
    monkeypatch.setattr(
        config_module.TrainingConfig,
        "from_yaml",
        classmethod(lambda cls, path, project_root=None: training),
    )
    monkeypatch.setattr(
        evaluation_module, "run_evaluation", lambda config: calls.append(config)
    )

    orchestrator.execute_stage("evaluate", tmp_path / "train.yaml", tmp_path)

    assert calls == [evaluation]


def test_execute_evaluate_rejects_disabled_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_pipeline.training import config as config_module

    monkeypatch.setattr(
        config_module.TrainingConfig,
        "from_yaml",
        classmethod(
            lambda cls, path, project_root=None: SimpleNamespace(evaluation=None)
        ),
    )

    with pytest.raises(ValueError, match="evaluation must be enabled"):
        orchestrator.execute_stage("evaluate", tmp_path / "train.yaml", tmp_path)


def test_execute_stage_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown pipeline stage"):
        orchestrator.execute_stage("export", tmp_path / "train.yaml", tmp_path)


def test_pipeline_without_evaluate_reports_no_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "s2"])
    run_dir = tmp_path / "runs" / "speaker"
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: run_dir)

    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: None,
    )

    assert outcome.cleaned is False


def test_successful_evaluation_pipeline_keeps_training_artifacts_for_human_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "evaluate"])
    run_dir = tmp_path / "runs" / "speaker"
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: run_dir)
    raw_checkpoint = run_dir / "training" / "s2" / "checkpoints" / "step.pt"
    raw_checkpoint.parent.mkdir(parents=True)
    raw_checkpoint.write_bytes(b"raw")

    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: None,
    )

    payload = json.loads(
        (run_dir / "pipeline-state.json").read_text(encoding="utf-8")
    )
    assert payload["stages"] == {
        "preprocess": "completed",
        "evaluate": "completed",
    }
    assert raw_checkpoint.is_file()
    assert outcome.cleaned is False


def test_stage_failure_preserves_training_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "evaluate"])
    run_dir = tmp_path / "runs" / "speaker"
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: run_dir)
    raw_checkpoint = run_dir / "training" / "s2" / "checkpoints" / "step.pt"
    raw_checkpoint.parent.mkdir(parents=True)
    raw_checkpoint.write_bytes(b"raw")

    with pytest.raises(PipelineStageError):
        run_pipeline(
            pipeline,
            tmp_path,
            execute_stage=lambda stage, config, root: (_ for _ in ()).throw(
                RuntimeError("stage failed")
            ),
        )
    assert raw_checkpoint.is_file()
