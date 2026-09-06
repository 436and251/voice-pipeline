from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from voice_pipeline.cli import evaluate as evaluate_cli
from voice_pipeline.cli.main import app
from voice_pipeline.common.errors import EvaluationError


runner = CliRunner()


def test_evaluate_cli_runs_enabled_evaluation(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("placeholder", encoding="utf-8")
    evaluation = object()
    monkeypatch.setattr(
        evaluate_cli.TrainingConfig,
        "from_yaml",
        lambda path, project_root: SimpleNamespace(evaluation=evaluation),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "run_evaluation",
        lambda value: SimpleNamespace(run_dir=tmp_path / "runs/speaker", shortlisted=("A", "B")),
    )

    result = runner.invoke(app, ["evaluate", "--config", str(config), "--project-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "shortlisted 2 candidates" in result.output


def test_evaluate_cli_rejects_disabled_evaluation(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        evaluate_cli.TrainingConfig,
        "from_yaml",
        lambda path, project_root: SimpleNamespace(evaluation=None),
    )

    result = runner.invoke(app, ["evaluate", "--config", str(config), "--project-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "evaluation must be enabled" in result.output


def test_evaluate_cli_converts_evaluation_failure_to_controlled_error(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        evaluate_cli.TrainingConfig,
        "from_yaml",
        lambda path, project_root: SimpleNamespace(evaluation=object()),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "run_evaluation",
        lambda value: (_ for _ in ()).throw(EvaluationError("no candidate passed")),
    )

    result = runner.invoke(app, ["evaluate", "--config", str(config), "--project-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Error: no candidate passed" in result.output
