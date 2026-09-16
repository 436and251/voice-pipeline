from __future__ import annotations

import json

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
        raise typer.UsageError("--json is required")
    typer.echo(json.dumps(build_descriptor(), ensure_ascii=False, separators=(",", ":")))


__all__ = ["app"]
