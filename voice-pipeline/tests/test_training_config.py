from pathlib import Path

import pytest
from typer.testing import CliRunner

from voice_pipeline.cli import train as train_cli
from voice_pipeline.cli.main import app
from voice_pipeline.training.config import TrainingConfig


runner = CliRunner()


def _write_config(tmp_path: Path, *, extra_s1: str = "", extra_s2: str = "") -> Path:
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""profile: {{name: v2ProPlus}}
experiment: {{name: speaker_001, output_root: runs}}
device: {{device: cpu, precision: fp32}}
s2:
  enabled: true
  batch_size: 2
  target_steps: 8
  checkpoint_every_steps: 4
  learning_rate: 0.0001
  text_low_lr_rate: 0.4
  freeze_quantizer: true
  grad_ckpt: false
  resume_from: resumes/s2.pt
{extra_s2}s1:
  enabled: true
  batch_size: 2
  gradient_accumulation: 4
  target_optimizer_steps: 6
  checkpoint_every_steps: 3
  resume_from: resumes/s1.pt
{extra_s1}""",
        encoding="utf-8",
    )
    return config


def test_training_config_maps_readme_yaml_to_existing_trainers(tmp_path: Path) -> None:
    config = TrainingConfig.from_yaml(_write_config(tmp_path), project_root=tmp_path)

    assert config.profile.name == "v2ProPlus"
    assert config.s2 is not None and config.s1 is not None
    assert config.s2.preprocess_dir == tmp_path / "runs/speaker_001/preprocess"
    assert config.s2.output_dir == tmp_path / "runs/speaker_001"
    assert config.s2.base_s2g_path == tmp_path / "models/pretrained/v2proplus/s2/s2Gv2ProPlus.pth"
    assert config.s2.target_optimizer_steps == 8
    assert config.s1.base_s1_path == tmp_path / "models/pretrained/v2proplus/s1/s1v3.ckpt"
    assert config.s1.target_optimizer_steps == 6
    assert config.s2_resume_from == tmp_path / "resumes/s2.pt"
    assert config.s1_resume_from == tmp_path / "resumes/s1.pt"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("profile: {name: v4}", "v2ProPlus"),
        ("target_steps: 0", "target_steps"),
        ("gradient_accumulation: 2", "gradient_accumulation"),
        ("text_low_lr_rate: 1.0", "text_low_lr_rate"),
        ("freeze_quantizer: false", "freeze_quantizer"),
        ("mystery: true", "unknown s1"),
    ],
)
def test_training_config_rejects_invalid_or_architecture_changing_values(
    tmp_path: Path, replacement: str, message: str
) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    if replacement.startswith("profile"):
        text = text.replace("profile: {name: v2ProPlus}", replacement)
    elif replacement.startswith("target_steps"):
        text = text.replace("target_steps: 8", replacement)
    elif replacement.startswith("gradient"):
        text = text.replace("gradient_accumulation: 4", replacement)
    elif replacement.startswith("text_low"):
        text = text.replace("text_low_lr_rate: 0.4", replacement)
    elif replacement.startswith("freeze"):
        text = text.replace("freeze_quantizer: true", replacement)
    else:
        text += f"  {replacement}\n"
    path.write_text(text, encoding="utf-8")

    with pytest.raises((KeyError, ValueError), match=message):
        TrainingConfig.from_yaml(path, project_root=tmp_path)


def test_train_all_runs_s2_then_s1_with_independent_resume_paths(tmp_path: Path, monkeypatch) -> None:
    path = _write_config(tmp_path)
    calls = []

    class FakeS2Trainer:
        @classmethod
        def from_pretrained(cls, config, *, resume_from=None):
            calls.append(("build_s2", config.target_optimizer_steps, resume_from))
            return cls()

        def train(self):
            calls.append(("train_s2",))
            return object()

    class FakeS1Trainer:
        @classmethod
        def from_pretrained(cls, config, *, resume_from=None):
            calls.append(("build_s1", config.target_optimizer_steps, resume_from))
            return cls()

        def train(self):
            calls.append(("train_s1",))
            return object()

    monkeypatch.setattr(train_cli, "S2Trainer", FakeS2Trainer)
    monkeypatch.setattr(train_cli, "S1Trainer", FakeS1Trainer)

    result = runner.invoke(app, ["train", "all", "-c", str(path), "--project-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls == [
        ("build_s2", 8, tmp_path / "resumes/s2.pt"),
        ("train_s2",),
        ("build_s1", 6, tmp_path / "resumes/s1.pt"),
        ("train_s1",),
    ]


def test_explicit_disabled_stage_fails_before_constructing_trainer(tmp_path: Path, monkeypatch) -> None:
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("s1:\n  enabled: true", "s1:\n  enabled: false"), encoding="utf-8")
    monkeypatch.setattr(
        train_cli.S1Trainer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("disabled S1 must not be constructed"),
    )

    result = runner.invoke(app, ["train", "s1", "-c", str(path), "--project-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "disabled" in result.output.lower()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("dataset: 123\n", "dataset must be a mapping"),
        ("evaluation: {typo: true}\n", "unknown evaluation"),
        ("1: value\n", "keys must be strings"),
    ],
)
def test_training_config_strictly_validates_shared_yaml_sections(
    tmp_path: Path, extra: str, message: str
) -> None:
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TrainingConfig.from_yaml(path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("learning_rate: 0.0001", "learning_rate: .nan", "finite"),
        ("learning_rate: 0.0001", "learning_rate: 0", "learning_rate"),
        ("text_low_lr_rate: 0.4", "text_low_lr_rate: .inf", "finite"),
        ("checkpoint_every_steps: 4", "checkpoint_every_steps: -1", "positive"),
    ],
)
def test_training_config_rejects_nonfinite_and_out_of_range_numbers(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TrainingConfig.from_yaml(path, project_root=tmp_path)


def test_training_cli_converts_nonstring_yaml_key_to_controlled_error(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "null: value\n", encoding="utf-8")

    result = runner.invoke(app, ["train", "all", "-c", str(path), "--project-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "keys must be strings" in result.output
