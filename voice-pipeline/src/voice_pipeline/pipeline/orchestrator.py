from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from voice_pipeline.common.errors import PipelineStageError
from voice_pipeline.pipeline.config import PipelineSpec
from voice_pipeline.pipeline.state import PipelineState


StageExecutor = Callable[[str, Path, Path], None]


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    executed: tuple[str, ...]
    skipped: tuple[str, ...]
    state_path: Path
    cleaned: bool


def execute_stage(stage: str, training_config: Path, project_root: Path) -> None:
    if stage == "preprocess":
        from voice_pipeline.training.manifest import read_manifest_records
        from voice_pipeline.training.preprocess.config import PreprocessConfig
        from voice_pipeline.training.preprocess.factory import build_preprocess_pipeline
        from voice_pipeline.training.preprocess.indexes import publish_training_indexes

        config = PreprocessConfig.from_yaml(training_config, project_root)
        manifest = read_manifest_records(config.manifest)
        pipeline = build_preprocess_pipeline(config)
        summary = pipeline.run(manifest.records, manifest.issues)
        publish_training_indexes(
            pipeline.context.preprocess_dir,
            manifest.records,
            set(summary.valid_sample_ids),
        )
        return

    if stage in {"s2", "s1", "evaluate"}:
        from voice_pipeline.training.config import TrainingConfig

        config = TrainingConfig.from_yaml(training_config, project_root)
        if stage == "s2":
            if config.s2 is None:
                raise ValueError("S2 training is disabled")
            from voice_pipeline.training.s2 import S2Trainer

            S2Trainer.from_pretrained(
                config.s2, resume_from=config.s2_resume_from
            ).train()
            return
        if stage == "s1":
            if config.s1 is None:
                raise ValueError("S1 training is disabled")
            from voice_pipeline.training.s1 import S1Trainer

            S1Trainer.from_pretrained(
                config.s1, resume_from=config.s1_resume_from
            ).train()
            return
        if config.evaluation is None:
            raise ValueError("evaluation must be enabled")
        from voice_pipeline.evaluation.pipeline import run_evaluation

        run_evaluation(config.evaluation)
        return

    raise ValueError(f"unknown pipeline stage: {stage}")


def _run_dir(training_config: Path, project_root: Path) -> Path:
    from voice_pipeline.training.config import TrainingConfig

    config = TrainingConfig.from_yaml(training_config, project_root)
    return (config.output_root / config.experiment_name).resolve()


def run_pipeline(
    path: Path,
    project_root: Path,
    *,
    execute_stage: StageExecutor = execute_stage,
) -> PipelineOutcome:
    root = Path(project_root).resolve()
    spec = PipelineSpec.load(path, root)
    state_path = _run_dir(spec.training_config, root) / "pipeline-state.json"
    state = PipelineState.load_or_create(state_path, spec)
    executed: list[str] = []
    skipped: list[str] = []

    for stage in spec.stages:
        if state.status(stage) == "completed":
            skipped.append(stage)
            continue
        state.start(stage)
        try:
            execute_stage(stage, spec.training_config, root)
        except Exception as error:
            state.fail(stage, error)
            raise PipelineStageError(f"stage {stage} failed: {error}") from error
        state.complete(stage)
        executed.append(stage)

    return PipelineOutcome(tuple(executed), tuple(skipped), state.path, False)


__all__ = [
    "PipelineOutcome",
    "PipelineStageError",
    "execute_stage",
    "run_pipeline",
]
