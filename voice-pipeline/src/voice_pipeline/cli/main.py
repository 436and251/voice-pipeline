import typer
from voice_pipeline import __version__
from voice_pipeline.cli.models import app as models_app
from voice_pipeline.cli.preprocess import app as preprocess_app

app = typer.Typer(name="voice-pipeline", help="GPT-SoVITS voice training and inference pipeline.")
app.add_typer(models_app, name="models")
app.add_typer(preprocess_app, name="preprocess")


@app.callback()
def main() -> None:
    """GPT-SoVITS voice training and inference pipeline."""


@app.command()
def version() -> None:
    """Print the framework version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
