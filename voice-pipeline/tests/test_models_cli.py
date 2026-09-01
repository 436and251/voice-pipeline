from pathlib import Path
from typer.testing import CliRunner

from voice_pipeline.cli.main import app

runner = CliRunner()


def test_models_verify_reports_missing_assets(tmp_path: Path):
    result = runner.invoke(app, ["models", "verify", "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "MISSING" in result.stdout
    assert "s1" in result.stdout
