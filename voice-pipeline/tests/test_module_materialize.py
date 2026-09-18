from __future__ import annotations

from pathlib import Path
import wave

import yaml

from voice_pipeline.module_api.job import ModuleJob, ModuleReference, ModuleTrainingData
from voice_pipeline.module_api.materialize import _reference_audio_usable, materialize_job
from voice_pipeline.pipeline.config import PipelineSpec
from voice_pipeline.training.config import TrainingConfig


def _job(
    tmp_path: Path,
    *,
    stages: tuple[str, ...] = ("preprocess", "s2", "s1"),
    explicit_reference: bool = True,
) -> ModuleJob:
    project_root = tmp_path / "project"
    dataset_list = project_root / "dataset" / "dataset.list"
    dataset_list.parent.mkdir(parents=True)
    (dataset_list.parent / "clip.wav").write_bytes(b"clip")
    dataset_list.write_text("clip.wav|speaker|ja|テスト\n", encoding="utf-8")
    reference_audio = project_root / "reference.wav"
    reference_audio.write_bytes(b"reference")
    for language in ("zh", "ja", "en", "mixed"):
        suite = project_root / "configs" / "eval" / f"{language}.txt"
        suite.parent.mkdir(parents=True, exist_ok=True)
        suite.write_text(f"{language} sample\n", encoding="utf-8")
    output_root = project_root / "runs"
    output_root.mkdir()
    job_dir = project_root / "jobs" / "job-001"
    job_dir.mkdir(parents=True)
    return ModuleJob(
        job_id="job-001",
        module_id="gpt-sovits-v2proplus",
        project_name="Acane",
        project_root=project_root.resolve(),
        output_root=output_root.resolve(),
        training_data=ModuleTrainingData(dataset_list.resolve(), "file"),
        framework="v2ProPlus",
        stages=stages,
        device="cuda:0",
        precision="fp16",
        parameters={
            "preprocess.resume": False,
            "s2.batch_size": 3,
            "s2.target_steps": 12,
            "s2.learning_rate": 0.0002,
            "s1.batch_size": 4,
            "s1.target_optimizer_steps": 10,
        },
        reference=(
            ModuleReference(reference_audio.resolve(), "参照音声です。", "ja")
            if "evaluate" in stages and explicit_reference
            else None
        ),
        job_dir=job_dir.resolve(),
    )


def test_materialize_job_writes_exact_training_and_pipeline_snapshots(tmp_path: Path):
    job = _job(tmp_path)

    result = materialize_job(job)

    training = yaml.safe_load(result.training_config.read_text(encoding="utf-8"))
    assert training == {
        "profile": {"name": "v2ProPlus"},
        "experiment": {"name": "Acane", "output_root": str(job.output_root)},
        "device": {"device": "cuda:0", "precision": "fp16"},
        "dataset": {"manifest": str(job.training_data.path)},
        "objective": {
            "training_languages": ["ja"],
            "target_languages": ["zh", "ja", "en"],
            "cross_language_preservation": "strict",
        },
        "preprocess": {"resume": False},
        "s2": {
            "enabled": True,
            "batch_size": 3,
            "target_steps": 12,
            "checkpoint_every_steps": 200,
            "learning_rate": 0.0002,
            "text_low_lr_rate": 0.4,
            "freeze_quantizer": True,
            "grad_ckpt": False,
        },
        "s1": {
            "enabled": True,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "target_optimizer_steps": 10,
            "checkpoint_every_steps": 100,
        },
        "evaluation": {"enabled": False},
    }
    pipeline = yaml.safe_load(result.pipeline_spec.read_text(encoding="utf-8"))
    assert pipeline == {
        "schema_version": 1,
        "config": "jobs/job-001/module/train.yaml",
        "stages": ["preprocess", "s2", "s1"],
    }
    assert result.run_dir == job.output_root / job.project_name
    assert PipelineSpec.load(result.pipeline_spec, job.project_root).training_config == result.training_config
    parsed = TrainingConfig.from_yaml(result.training_config, job.project_root)
    assert parsed.s2 is not None and parsed.s1 is not None


def test_materialize_job_marks_unselected_training_stage_disabled(tmp_path: Path):
    job = _job(tmp_path, stages=("preprocess", "s2"))

    result = materialize_job(job)
    training = yaml.safe_load(result.training_config.read_text(encoding="utf-8"))

    assert training["s2"]["enabled"] is True
    assert training["s1"] == {"enabled": False}
    assert training["evaluation"] == {"enabled": False}


def test_materialize_job_binds_evaluation_to_target_reference(tmp_path: Path):
    job = _job(tmp_path, stages=("preprocess", "s2", "s1", "evaluate"))

    result = materialize_job(job)
    training = yaml.safe_load(result.training_config.read_text(encoding="utf-8"))

    assert training["experiment"]["name"] == "Acane"
    assert training["objective"] == {
        "training_languages": ["ja"],
        "target_languages": ["zh", "ja", "en"],
        "cross_language_preservation": "strict",
    }
    assert training["evaluation"]["reference"] == {
        "audio": str(job.reference.audio),
        "text": "参照音声です。",
        "language": "ja",
    }
    assert training["evaluation"]["speaker_references"] == [str(job.reference.audio)]
    parsed = TrainingConfig.from_yaml(result.training_config, job.project_root)
    assert parsed.evaluation is not None
    assert parsed.evaluation.model_name == "Acane"
    assert parsed.evaluation.reference.audio == job.reference.audio


def test_materialize_job_uses_audio_miner_huggingface_cache(
    tmp_path: Path, monkeypatch
):
    hf_home = tmp_path / "shared-models" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    job = _job(tmp_path, stages=("preprocess", "s2", "s1", "evaluate"))

    result = materialize_job(job)
    training = yaml.safe_load(result.training_config.read_text(encoding="utf-8"))

    assert training["evaluation"]["models"]["cache_dir"] == str(
        (hf_home / "hub").resolve()
    )


def test_materialize_job_selects_first_usable_evaluation_reference_from_training_data(
    tmp_path: Path, monkeypatch
):
    job = _job(
        tmp_path,
        stages=("preprocess", "s2", "s1", "evaluate"),
        explicit_reference=False,
    )
    bad_audio = job.training_data.path.parent / "bad.wav"
    bad_audio.write_bytes(b"bad")
    job.training_data.path.write_text(
        "bad.wav|speaker|ja|不正\nclip.wav|speaker|ja|テスト\n",
        encoding="utf-8",
    )
    checked = []

    def usable(audio: Path):
        checked.append(audio.name)
        return audio != bad_audio

    monkeypatch.setattr(
        "voice_pipeline.module_api.materialize._reference_audio_usable", usable
    )

    result = materialize_job(job)
    training = yaml.safe_load(result.training_config.read_text(encoding="utf-8"))
    expected_audio = (job.training_data.path.parent / "clip.wav").resolve()

    assert training["evaluation"]["reference"] == {
        "audio": str(expected_audio),
        "text": "テスト",
        "language": "ja",
    }
    assert training["evaluation"]["speaker_references"] == [str(expected_audio)]
    assert checked == ["bad.wav", "clip.wav"]


def test_materialize_job_is_byte_deterministic_and_leaves_no_temporary_files(tmp_path: Path):
    job = _job(tmp_path)

    first = materialize_job(job)
    first_training = first.training_config.read_bytes()
    first_pipeline = first.pipeline_spec.read_bytes()
    second = materialize_job(job)

    assert second == first
    assert second.training_config.read_bytes() == first_training
    assert second.pipeline_spec.read_bytes() == first_pipeline
    assert sorted(path.name for path in second.training_config.parent.iterdir()) == [
        "pipeline.yaml",
        "train.yaml",
    ]


def test_reference_audio_probe_checks_decode_and_duration(tmp_path: Path):
    valid = tmp_path / "valid.wav"
    too_short = tmp_path / "short.wav"
    corrupt = tmp_path / "corrupt.wav"
    for path, seconds in ((valid, 3), (too_short, 2)):
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(32_000)
            stream.writeframes(b"\0\0" * (32_000 * seconds))
    corrupt.write_bytes(b"not audio")

    assert _reference_audio_usable(valid)
    assert not _reference_audio_usable(too_short)
    assert not _reference_audio_usable(corrupt)
