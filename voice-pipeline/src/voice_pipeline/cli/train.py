from pathlib import Path

import typer

from voice_pipeline.training.config import TrainingConfig
from voice_pipeline.training.s1 import S1Trainer
from voice_pipeline.training.s2 import S2Trainer


app = typer.Typer(help="Train GPT-SoVITS v2ProPlus S1 and S2 models.")


def _load(config_path: Path, project_root: Path) -> TrainingConfig:
    try:
        return TrainingConfig.from_yaml(config_path, project_root=project_root)
    except (KeyError, OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _run_s1(config: TrainingConfig) -> None:
    if config.s1 is None:
        typer.echo("Error: S1 training is disabled", err=True)
        raise typer.Exit(code=1)
    S1Trainer.from_pretrained(config.s1, resume_from=config.s1_resume_from).train()
    typer.echo("S1 training complete")


def _run_s2(config: TrainingConfig) -> None:
    if config.s2 is None:
        typer.echo("Error: S2 training is disabled", err=True)
        raise typer.Exit(code=1)
    S2Trainer.from_pretrained(config.s2, resume_from=config.s2_resume_from).train()
    typer.echo("S2 training complete")


def _execute(stage: str, config_path: Path, project_root: Path) -> None:
    config = _load(config_path, project_root)
    try:
        if stage == "s2":
            _run_s2(config)
        elif stage == "s1":
            _run_s1(config)
        else:
            if config.s2 is not None:
                _run_s2(config)
            if config.s1 is not None:
                _run_s1(config)
    except typer.Exit:
        raise
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("s1")
def train_s1(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    project_root: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
) -> None:
    _execute("s1", config, project_root)


@app.command("s2")
def train_s2(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    project_root: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
) -> None:
    _execute("s2", config, project_root)


@app.command("all")
def train_all(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    project_root: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
) -> None:
    _execute("all", config, project_root)
