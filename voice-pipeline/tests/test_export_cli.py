from pathlib import Path

from typer.testing import CliRunner

from voice_pipeline.cli import export as export_cli
from voice_pipeline.cli.main import app


runner = CliRunner()


def test_export_command_converts_all_without_implicit_selection(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    shortlist = object()
    calls = []
    monkeypatch.setattr(export_cli.Shortlist, "load", lambda run_dir, project_root: shortlist)
    monkeypatch.setattr(
        export_cli,
        "export_candidates",
        lambda loaded, run_dir, project_root, overwrite=False: calls.append((loaded, run_dir, project_root, overwrite))
        or [run / "export" / "candidates" / "candidate_A", run / "export" / "candidates" / "candidate_B"],
    )
    monkeypatch.setattr(export_cli, "promote_candidate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not promote")))

    result = runner.invoke(app, ["export", "--run", str(run), "--project-root", str(tmp_path)])

    assert result.exit_code == 0
    assert "exported 2 candidates" in result.stdout
    assert calls == [(shortlist, run.resolve(), tmp_path.resolve(), False)]


def test_export_command_promotes_only_explicit_human_selection(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    model_root = tmp_path / "published"
    calls = []
    monkeypatch.setattr(export_cli.Shortlist, "load", lambda *args: (_ for _ in ()).throw(AssertionError("must not reload shortlist")))
    monkeypatch.setattr(export_cli, "export_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not convert")))
    monkeypatch.setattr(
        export_cli,
        "promote_candidate",
        lambda run_dir, candidate_id, project_root, overwrite=False, model_root=None: calls.append(
            (run_dir, candidate_id, project_root, overwrite, model_root)
        )
        or model_root / "speaker",
    )

    result = runner.invoke(
        app,
        [
            "export", "--run", str(run), "--project-root", str(tmp_path),
            "--select", "candidate_B", "--model-root", str(model_root), "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert "promoted candidate_B" in result.stdout
    assert calls == [(run.resolve(), "candidate_B", tmp_path.resolve(), True, model_root.resolve())]


def test_export_command_reports_unknown_candidate_as_controlled_error(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(export_cli, "promote_candidate", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("candidate does not exist: candidate_Z")))
    result = runner.invoke(app, ["export", "--run", str(run), "--project-root", str(tmp_path), "--select", "candidate_Z"])
    assert result.exit_code == 1
    assert "Error: candidate does not exist: candidate_Z" in result.stderr


def test_export_help_does_not_offer_automatic_selection():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "--select" in result.stdout
    assert "best" not in result.stdout.lower()


def test_model_root_requires_explicit_selection(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    result = runner.invoke(app, ["export", "--run", str(run), "--project-root", str(tmp_path), "--model-root", str(tmp_path / "models")])
    assert result.exit_code == 1
    assert "--model-root requires --select" in result.stderr
