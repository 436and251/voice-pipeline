from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from voice_pipeline.cli import run as run_cli
from voice_pipeline.cli.main import app
from voice_pipeline.common.errors import PipelineStageError


runner = CliRunner()


def test_run_cli_reports_executed_skipped_state_and_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("placeholder", encoding="utf-8")
    state_path = tmp_path / "runs" / "speaker" / "pipeline-state.json"
    monkeypatch.setattr(
        run_cli,
        "run_pipeline",
        lambda path, root: SimpleNamespace(
            executed=("s2", "s1", "evaluate"),
            skipped=("preprocess",),
            state_path=state_path,
            cleaned=True,
        ),
    )

    result = runner.invoke(
        app, ["run", str(pipeline), "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "executed=s2,s1,evaluate" in result.output
    assert "skipped=preprocess" in result.output
    assert f"state={state_path}" in result.output
    assert "cleaned=yes" in result.output


def test_run_cli_returns_one_for_pipeline_failure(
    tmp_path: Path, monkeypatch
) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("placeholder", encoding="utf-8")

    def fail(path: Path, root: Path):
        raise PipelineStageError("stage s2 failed")

    monkeypatch.setattr(run_cli, "run_pipeline", fail)

    result = runner.invoke(
        app, ["run", str(pipeline), "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "Error: stage s2 failed" in result.output
