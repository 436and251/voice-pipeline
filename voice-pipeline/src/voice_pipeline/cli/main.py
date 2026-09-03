import typer
from voice_pipeline import __version__
from voice_pipeline.cli.models import app as models_app
from voice_pipeline.cli.preprocess import app as preprocess_app
from voice_pipeline.cli.train import app as train_app
from voice_pipeline.cli.export import export_command
from voice_pipeline.cli.infer import app as infer_app

app = typer.Typer(name="voice-pipeline", help="GPT-SoVITS voice training and inference pipeline.")
app.add_typer(models_app, name="models")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(train_app, name="train")
app.command("export")(export_command)
app.add_typer(infer_app, name="infer")


@app.callback()
def main() -> None:
    """GPT-SoVITS voice training and inference pipeline."""


@app.command()
def version() -> None:
    """Print the framework version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
