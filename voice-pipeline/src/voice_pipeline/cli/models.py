from pathlib import Path
import typer

from voice_pipeline.common.assets import verify_profile_assets
from voice_pipeline.profiles.registry import ProfileRegistry

app = typer.Typer(help="Inspect and verify model assets.")


@app.command()
def verify(
    project_root: Path = typer.Option(Path.cwd(), exists=True, file_okay=False, dir_okay=True),
    profile: str = typer.Option("v2ProPlus"),
) -> None:
    checks = verify_profile_assets(ProfileRegistry.get(profile), project_root)
    missing = False
    for check in checks:
        status = "OK" if check.exists else "MISSING"
        typer.echo(f"{status:7} {check.name:8} {check.path}")
        missing |= not check.exists
    if missing:
        raise typer.Exit(code=1)
