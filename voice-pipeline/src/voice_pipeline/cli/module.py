from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

from voice_pipeline.module_api.descriptor import build_descriptor


app = typer.Typer(help="Machine-readable training-module interface.")


@app.callback()
def main() -> None:
    """Machine-readable training-module interface."""


@app.command("describe")
def describe(
    as_json: bool = typer.Option(False, "--json", help="Emit one compact JSON object."),
) -> None:
    """Describe the module without loading models or training dependencies."""
    if not as_json:
        typer.echo("Error: --json is required", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(build_descriptor(), ensure_ascii=False, separators=(",", ":")))


@app.command("run")
def run(
    job: Path = typer.Option(..., "--job", exists=True, dir_okay=False),
    events_jsonl: bool = typer.Option(False, "--events-jsonl"),
) -> None:
    """Run one validated module job using the JSONL event protocol."""
    if not events_jsonl:
        typer.echo("Error: --events-jsonl is required", err=True)
        raise typer.Exit(code=2)
    from voice_pipeline.module_api.runner import run_module_job

    try:
        run_module_job(job.resolve(), sys.stdout, diagnostics=sys.stderr)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("promote")
def promote(
    job: Path = typer.Option(..., "--job", exists=True, dir_okay=False),
    selection: str = typer.Option(..., "--selection"),
    events_jsonl: bool = typer.Option(False, "--events-jsonl"),
) -> None:
    """Promote one explicit human-selected candidate."""
    if not events_jsonl:
        typer.echo("Error: --events-jsonl is required", err=True)
        raise typer.Exit(code=2)
    from voice_pipeline.module_api.promote import promote_module_candidate

    try:
        promote_module_candidate(
            job.resolve(), selection, sys.stdout, diagnostics=sys.stderr
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("infer")
def infer(
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    events_jsonl: bool = typer.Option(False, "--events-jsonl"),
) -> None:
    """Run inference with the latest human-promoted model."""
    if not events_jsonl:
        typer.echo("Error: --events-jsonl is required", err=True)
        raise typer.Exit(code=2)
    from voice_pipeline.module_api.infer import run_module_inference

    try:
        run_module_inference(
            request.resolve(), sys.stdout, diagnostics=sys.stderr
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


__all__ = ["app"]
