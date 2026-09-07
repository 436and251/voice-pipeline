from pathlib import Path

import typer

from voice_pipeline.common.errors import VoicePipelineError
from voice_pipeline.pipeline.orchestrator import run_pipeline


def run_command(
    pipeline: Path = typer.Argument(..., exists=True, dir_okay=False),
    project_root: Path = typer.Option(
        Path.cwd(), "--project-root", exists=True, file_okay=False
    ),
) -> None:
    """Run declared training and evaluation stages with persistent state."""
    try:
        outcome = run_pipeline(pipeline.resolve(), project_root.resolve())
    except (FileNotFoundError, OSError, RuntimeError, ValueError, VoicePipelineError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    executed = ",".join(outcome.executed) or "-"
    skipped = ",".join(outcome.skipped) or "-"
    typer.echo(
        f"executed={executed} skipped={skipped} state={outcome.state_path} "
        f"cleaned={'yes' if outcome.cleaned else 'no'}"
    )


__all__ = ["run_command"]
