from pathlib import Path

import typer

from voice_pipeline.training.manifest import read_manifest_records
from voice_pipeline.training.preprocess.config import PreprocessConfig
from voice_pipeline.training.preprocess.factory import DEPENDENCIES, build_preprocess_pipeline
from voice_pipeline.training.preprocess.indexes import publish_training_indexes


app = typer.Typer(help="Prepare official GPT-SoVITS v2ProPlus training artifacts.")


def _execute(config_path: Path, selected_stage: str | None) -> None:
    if selected_stage is not None and selected_stage not in DEPENDENCIES:
        raise typer.BadParameter(f"unknown preprocessing stage: {selected_stage}")
    try:
        config = PreprocessConfig.from_yaml(config_path)
        manifest = read_manifest_records(config.manifest)
        pipeline = build_preprocess_pipeline(config, selected_stage=selected_stage)
        summary = pipeline.run(manifest.records, manifest.issues, selected_stage=selected_stage)
        if selected_stage is None:
            publish_training_indexes(
                pipeline.context.preprocess_dir,
                manifest.records,
                set(summary.valid_sample_ids),
            )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    report = pipeline.context.preprocess_dir / "quarantine.jsonl"
    typer.echo(
        f"preprocess complete: bad={len(summary.quarantined)} "
        f"allowed={summary.allowed_bad} valid={len(summary.valid_sample_ids)} report={report}"
    )


@app.command("all")
def preprocess_all(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    _execute(config, None)


@app.command("stage")
def preprocess_stage(
    name: str,
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    _execute(config, name)
