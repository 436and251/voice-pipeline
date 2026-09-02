from pathlib import Path

import pytest

from voice_pipeline.training.preprocess.config import PreprocessConfig
from voice_pipeline.training.preprocess.factory import build_preprocess_pipeline
from voice_pipeline.training.preprocess import factory


def write_config(tmp_path, profile="v2ProPlus"):
    manifest = tmp_path / "data.list"
    manifest.write_text("", encoding="utf-8")
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        f"""profile:
  name: {profile}
experiment:
  name: speaker_001
  output_root: runs
device:
  device: cpu
  precision: fp16
dataset:
  manifest: data.list
preprocess:
  resume: true
""",
        encoding="utf-8",
    )
    return path


def test_config_reads_existing_readme_shape(tmp_path):
    config = PreprocessConfig.from_yaml(write_config(tmp_path), project_root=tmp_path)
    assert config.profile.name == "v2ProPlus"
    assert config.manifest.name == "data.list"
    assert config.manifest.is_absolute()
    assert config.output_root == tmp_path / "runs"
    assert config.precision == "fp16"
    assert config.resume is True


def test_config_rejects_non_v2proplus_profile_and_malformed_yaml(tmp_path):
    with pytest.raises(ValueError, match="v2ProPlus"):
        PreprocessConfig.from_yaml(write_config(tmp_path, "v4"), project_root=tmp_path)

    malformed = tmp_path / "bad.yaml"
    malformed.write_text("profile: [", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        PreprocessConfig.from_yaml(malformed, project_root=tmp_path)


def test_missing_required_asset_fails_before_experiment_is_created(tmp_path):
    config = PreprocessConfig.from_yaml(write_config(tmp_path), project_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="bert"):
        build_preprocess_pipeline(config, selected_stage="text")
    assert not (tmp_path / "runs").exists()


def test_text_target_does_not_require_hubert_s2g_or_speaker(monkeypatch, tmp_path):
    config = PreprocessConfig.from_yaml(write_config(tmp_path), project_root=tmp_path)
    for relative in (
        config.profile.bert_relative_path,
        config.profile.g2pw_relative_path,
        config.profile.nltk_data_relative_path,
        config.profile.langdetect_relative_path,
    ):
        (tmp_path / relative).mkdir(parents=True)
    monkeypatch.setattr(factory, "MultilingualFrontend", lambda *args: object())

    pipeline = build_preprocess_pipeline(config, selected_stage="text")
    assert set(pipeline.stages) == {"text"}
