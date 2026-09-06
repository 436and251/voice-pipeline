from pathlib import Path

import typer

from voice_pipeline.common.errors import VoicePipelineError
from voice_pipeline.evaluation.pipeline import run_evaluation
from voice_pipeline.training.config import TrainingConfig


def evaluate_command(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    project_root: Path = typer.Option(Path.cwd(), "--project-root", exists=True, file_okay=False),
) -> None:
    """Evaluate checkpoints and export an anonymous human-listening shortlist."""
    try:
        training = TrainingConfig.from_yaml(config.resolve(), project_root.resolve())
        if training.evaluation is None:
            raise ValueError("evaluation must be enabled")
        outcome = run_evaluation(training.evaluation)
        typer.echo(
            f"shortlisted {len(outcome.shortlisted)} candidates in "
            f"{outcome.run_dir / 'evaluation'}"
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError, VoicePipelineError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


__all__ = ["evaluate_command"]
