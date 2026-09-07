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
        "cleanup_successful_run",
        lambda run_dir, project_root: calls.append(("cleanup", run_dir, project_root)),
        raising=False,
    )
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
    assert calls == [
        (run.resolve(), "candidate_B", tmp_path.resolve(), True, model_root.resolve()),
        ("cleanup", run.resolve(), tmp_path.resolve()),
    ]


def test_failed_promotion_never_cleans(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(
        export_cli,
        "promote_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("promotion failed")),
    )
    monkeypatch.setattr(
        export_cli,
        "cleanup_successful_run",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not clean")),
        raising=False,
    )

    result = runner.invoke(
        app,
        ["export", "--run", str(run), "--project-root", str(tmp_path), "--select", "candidate_A"],
    )

    assert result.exit_code == 1
    assert "promotion failed" in result.stderr


def test_final_model_root_must_be_outside_training_run(tmp_path: Path, monkeypatch):
    run = tmp_path / "runs" / "speaker"
    run.mkdir(parents=True)
    monkeypatch.setattr(
        export_cli,
        "promote_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not promote")),
    )

    result = runner.invoke(
        app,
        [
            "export",
            "--run",
            str(run),
            "--project-root",
            str(tmp_path),
            "--select",
            "candidate_A",
            "--model-root",
            str(run / "final"),
        ],
    )

    assert result.exit_code == 1
    assert "outside the training run" in result.stderr


def test_cleanup_failure_reports_that_promoted_model_is_preserved(
    tmp_path: Path, monkeypatch
):
    run = tmp_path / "run"
    run.mkdir()
    promoted = tmp_path / "models" / "speaker"

    def promote(*args, **kwargs):
        promoted.mkdir(parents=True)
        (promoted / "model.yaml").write_text("preserved", encoding="utf-8")
        return promoted

    monkeypatch.setattr(export_cli, "promote_candidate", promote)
    monkeypatch.setattr(
        export_cli,
        "cleanup_successful_run",
        lambda *args: (_ for _ in ()).throw(ValueError("validation failed")),
        raising=False,
    )

    result = runner.invoke(
        app,
        ["export", "--run", str(run), "--project-root", str(tmp_path), "--select", "candidate_A"],
    )

    assert result.exit_code == 1
    assert "promoted candidate_A" in result.stderr
    assert "cleanup failed: validation failed" in result.stderr
    assert (promoted / "model.yaml").read_text(encoding="utf-8") == "preserved"


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
