from pathlib import Path

import typer

from voice_pipeline.common.model_bundle import Shortlist
from voice_pipeline.exporting.bundles import export_candidates, promote_candidate


def export_command(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False, help="Training run containing evaluation/shortlist.yaml."),
    project_root: Path = typer.Option(Path.cwd(), "--project-root", exists=True, file_okay=False),
    select: str | None = typer.Option(None, "--select", help="Human-selected exported candidate ID to promote."),
    model_root: Path | None = typer.Option(None, "--model-root", file_okay=False, help="Final model directory root (default: PROJECT_ROOT/models)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Explicitly replace an existing export or final model."),
) -> None:
    """Export every shortlisted candidate, or promote one explicit human selection."""
    run = run.resolve()
    project_root = project_root.resolve()
    model_root = model_root.resolve() if model_root is not None else None
    try:
        if select is None and model_root is not None:
            raise ValueError("--model-root requires --select")
        if select is None:
            shortlist = Shortlist.load(run, project_root)
            exported = export_candidates(shortlist, run, project_root, overwrite=overwrite)
            typer.echo(f"exported {len(exported)} candidates to {run / 'export' / 'candidates'}")
        else:
            promoted = promote_candidate(
                run,
                select,
                project_root,
                overwrite=overwrite,
                model_root=model_root,
            )
            typer.echo(f"promoted {select} to {promoted}")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


__all__ = ["export_command"]
