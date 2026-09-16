from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from voice_pipeline.common.errors import PipelineStageError
from voice_pipeline.module_api.cancellation import FileCancellationToken, ModuleCancelled
from voice_pipeline.pipeline.config import PipelineSpec
from voice_pipeline.pipeline.state import PipelineState


StageExecutor = Callable[[str, Path, Path], None]
EventSink = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    executed: tuple[str, ...]
    skipped: tuple[str, ...]
    state_path: Path
    cleaned: bool


def execute_stage(
    stage: str,
    training_config: Path,
    project_root: Path,
    *,
    event_sink: EventSink | None = None,
    cancellation: FileCancellationToken | None = None,
) -> None:
    if stage == "preprocess":
        from voice_pipeline.training.manifest import read_manifest_records
        from voice_pipeline.training.preprocess.config import PreprocessConfig
        from voice_pipeline.training.preprocess.factory import build_preprocess_pipeline
        from voice_pipeline.training.preprocess.indexes import publish_training_indexes

        config = PreprocessConfig.from_yaml(training_config, project_root)
        manifest = read_manifest_records(config.manifest)
        pipeline = build_preprocess_pipeline(config)
        run_options = {"cancellation": cancellation} if cancellation is not None else {}
        summary = pipeline.run(manifest.records, manifest.issues, **run_options)
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

            options = {"resume_from": config.s2_resume_from}
            if event_sink is not None:
                options["event_sink"] = event_sink
            if cancellation is not None:
                options["cancellation"] = cancellation
            S2Trainer.from_pretrained(config.s2, **options).train()
            return
        if stage == "s1":
            if config.s1 is None:
                raise ValueError("S1 training is disabled")
            from voice_pipeline.training.s1 import S1Trainer

            options = {"resume_from": config.s1_resume_from}
            if event_sink is not None:
                options["event_sink"] = event_sink
            if cancellation is not None:
                options["cancellation"] = cancellation
            S1Trainer.from_pretrained(config.s1, **options).train()
            return
        if config.evaluation is None:
            raise ValueError("evaluation must be enabled")
        from voice_pipeline.evaluation.pipeline import run_evaluation

        run_evaluation(config.evaluation)
        return

    raise ValueError(f"unknown pipeline stage: {stage}")


def _run_dir(training_config: Path, project_root: Path) -> Path:
    from voice_pipeline.training.preprocess.config import PreprocessConfig

    config = PreprocessConfig.from_yaml(training_config, project_root)
    return (config.output_root / config.experiment_name).resolve()


def run_pipeline(
    path: Path,
    project_root: Path,
    *,
    execute_stage: StageExecutor = execute_stage,
    event_sink: EventSink | None = None,
    cancellation: FileCancellationToken | None = None,
) -> PipelineOutcome:
    root = Path(project_root).resolve()
    spec = PipelineSpec.load(path, root)
    state_path = _run_dir(spec.training_config, root) / "pipeline-state.json"
    state = PipelineState.load_or_create(state_path, spec)
    executed: list[str] = []
    skipped: list[str] = []

    for stage in spec.stages:
        if cancellation is not None:
            cancellation.raise_if_requested()
        if state.status(stage) == "completed":
            skipped.append(stage)
            _emit(event_sink, "stage_cache_hit", stage)
            continue
        state.start(stage)
        _emit(event_sink, "stage_started", stage)
        try:
            if execute_stage is _DEFAULT_STAGE_EXECUTOR:
                execute_stage(
                    stage,
                    spec.training_config,
                    root,
                    event_sink=event_sink,
                    cancellation=cancellation,
                )
            else:
                execute_stage(stage, spec.training_config, root)
            if cancellation is not None:
                cancellation.raise_if_requested()
        except ModuleCancelled:
            _emit(event_sink, "stage_cancelled", stage)
            raise
        except Exception as error:
            state.fail(stage, error)
            _emit(
                event_sink,
                "stage_failed",
                stage,
                message_key="pipeline.stage_failed",
                message_args={"error": str(error)},
            )
            raise PipelineStageError(f"stage {stage} failed: {error}") from error
        state.complete(stage)
        executed.append(stage)
        _emit(event_sink, "stage_completed", stage)

    _emit(event_sink, "pipeline_completed")
    return PipelineOutcome(tuple(executed), tuple(skipped), state.path, False)


_DEFAULT_STAGE_EXECUTOR = execute_stage


def _emit(
    sink: EventSink | None,
    event_type: str,
    stage: str | None = None,
    **fields: object,
) -> None:
    if sink is None:
        return
    event: dict[str, object] = {"type": event_type, **fields}
    if stage is not None:
        event["stage"] = stage
    sink(event)


__all__ = [
    "PipelineOutcome",
    "PipelineStageError",
    "execute_stage",
    "run_pipeline",
]
