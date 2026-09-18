from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path

import pytest

from voice_pipeline.module_api.job import ModuleJob


@pytest.fixture()
def job_file(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    dataset_list = project_root / "dataset" / "dataset.list"
    dataset_list.parent.mkdir(parents=True)
    dataset_list.write_text("clip.wav|speaker|ja|テスト\n", encoding="utf-8")
    reference_audio = project_root / "reference.wav"
    reference_audio.write_bytes(b"reference")
    output_root = project_root / "runs"
    output_root.mkdir()
    job_dir = project_root / "jobs" / "job-001"
    job_dir.mkdir(parents=True)
    path = job_dir / "job.json"
    payload = {
        "protocol_version": 2,
        "job_id": "job-001",
        "module_id": "gpt-sovits-v2proplus",
        "project_name": "Acane",
        "project_root": str(project_root.resolve()),
        "output_root": str(output_root.resolve()),
        "training_data": {"path": str(dataset_list.resolve()), "kind": "file"},
        "framework": "v2ProPlus",
        "stages": ["preprocess", "s2", "s1", "evaluate"],
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {"s2.batch_size": 2, "preprocess.resume": True},
        "reference": {
            "audio": str(reference_audio.resolve()),
            "text": "参照音声です。",
            "language": "ja",
        },
        "job_dir": str(job_dir.resolve()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_job_loads_valid_contract_as_resolved_frozen_value(job_file: Path):
    job = ModuleJob.load(job_file)

    assert job.job_id == "job-001"
    assert job.project_name == "Acane"
    assert job.project_root == Path(_payload(job_file)["project_root"])
    assert job.module_id == "gpt-sovits-v2proplus"
    assert job.training_data.path == Path(_payload(job_file)["training_data"]["path"])
    assert job.training_data.kind == "file"
    assert job.stages == ("preprocess", "s2", "s1", "evaluate")
    assert job.parameters == {"s2.batch_size": 2, "preprocess.resume": True}
    assert job.reference is not None
    assert job.reference.language == "ja"
    with pytest.raises(FrozenInstanceError):
        job.job_id = "changed"


def test_job_strictly_validates_training_data_shape(job_file: Path):
    payload = _payload(job_file)
    payload["training_data"]["extra"] = True
    _write(job_file, payload)

    with pytest.raises(ValueError, match="training_data"):
        ModuleJob.load(job_file)


def test_job_rejects_output_escape(job_file: Path):
    payload = _payload(job_file)
    payload["output_root"] = str(job_file.parents[3] / "outside")
    _write(job_file, payload)

    with pytest.raises(ValueError, match="output_root"):
        ModuleJob.load(job_file)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", 1, "protocol_version"),
        ("protocol_version", True, "protocol_version"),
        ("protocol_version", 2.0, "protocol_version"),
        ("module_id", "other-module", "module_id"),
        ("framework", "unknown", "framework"),
        ("precision", "bf16", "precision"),
        ("stages", ["s1", "s2"], "stages"),
        ("stages", ["preprocess", "s2", "s2"], "stages"),
        ("stages", [], "stages"),
    ],
)
def test_job_rejects_invalid_protocol_choices_and_stage_order(
    job_file: Path, field: str, value: object, message: str
):
    payload = _payload(job_file)
    payload[field] = value
    _write(job_file, payload)

    with pytest.raises(ValueError, match=message):
        ModuleJob.load(job_file)


def test_job_rejects_unknown_top_level_and_parameter_fields(job_file: Path):
    payload = _payload(job_file)
    payload["surprise"] = True
    _write(job_file, payload)
    with pytest.raises(ValueError, match="unknown job field"):
        ModuleJob.load(job_file)

    payload.pop("surprise")
    payload["parameters"] = {"s2.magic": 7}
    _write(job_file, payload)
    with pytest.raises(ValueError, match="unknown parameter"):
        ModuleJob.load(job_file)


def test_job_rejects_missing_top_level_field(job_file: Path):
    payload = _payload(job_file)
    payload.pop("device")
    _write(job_file, payload)

    with pytest.raises(ValueError, match="missing job field: device"):
        ModuleJob.load(job_file)


@pytest.mark.parametrize("job_id", ["../escape", "job/escape", "different-job"])
def test_job_id_must_be_safe_and_match_its_directory(job_file: Path, job_id: str):
    payload = _payload(job_file)
    payload["job_id"] = job_id
    _write(job_file, payload)

    with pytest.raises(ValueError, match="job_id"):
        ModuleJob.load(job_file)


@pytest.mark.parametrize("project_name", ["", "Acane/other", "名字", True])
def test_job_rejects_unsafe_project_name(job_file: Path, project_name: object):
    payload = _payload(job_file)
    payload["project_name"] = project_name
    _write(job_file, payload)

    with pytest.raises(ValueError, match="project_name"):
        ModuleJob.load(job_file)


def test_job_rejects_boolean_as_integer_parameter(job_file: Path):
    payload = _payload(job_file)
    payload["parameters"] = {"s1.batch_size": True}
    _write(job_file, payload)

    with pytest.raises(ValueError, match="s1.batch_size.*integer"):
        ModuleJob.load(job_file)


def test_job_enforces_numeric_parameter_constraints(job_file: Path):
    payload = _payload(job_file)
    payload["parameters"] = {"s2.target_steps": 0}
    _write(job_file, payload)

    with pytest.raises(ValueError, match="s2.target_steps.*at least 1"):
        ModuleJob.load(job_file)


def test_job_enforces_candidate_label_limit(job_file: Path):
    payload = _payload(job_file)
    payload["parameters"] = {"evaluation.shortlist_size": 27}
    _write(job_file, payload)

    with pytest.raises(ValueError, match="evaluation.shortlist_size.*at most 26"):
        ModuleJob.load(job_file)


@pytest.mark.parametrize("value", [10**1000, float("nan"), float("inf")])
def test_job_rejects_non_finite_numeric_parameter_without_crashing(job_file: Path, value: float):
    payload = _payload(job_file)
    payload["parameters"] = {"s2.learning_rate": value}
    _write(job_file, payload)

    with pytest.raises(ValueError, match="s2.learning_rate.*finite"):
        ModuleJob.load(job_file)


@pytest.mark.parametrize("field", ["project_root", "output_root", "job_dir"])
def test_job_requires_absolute_paths(job_file: Path, field: str):
    payload = _payload(job_file)
    payload[field] = "relative/path"
    _write(job_file, payload)

    with pytest.raises(ValueError, match=field):
        ModuleJob.load(job_file)


def test_job_rejects_wrong_job_directory(job_file: Path):
    payload = _payload(job_file)
    payload["job_dir"] = str(job_file.parent.parent.resolve())
    _write(job_file, payload)

    with pytest.raises(ValueError, match="job_dir"):
        ModuleJob.load(job_file)


def test_job_accepts_dataset_outside_project_as_read_only_input(job_file: Path):
    outside = job_file.parents[3] / "outside.list"
    outside.write_text("outside", encoding="utf-8")
    payload = _payload(job_file)
    payload["training_data"]["path"] = str(outside.resolve())
    _write(job_file, payload)

    assert ModuleJob.load(job_file).training_data.path == outside.resolve()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.update(kind="directory"), "training_data"),
        (lambda data: data.update(path="relative.list"), "training_data.path"),
    ],
)
def test_job_rejects_training_data_that_disagrees_with_descriptor(
    job_file: Path, change, message: str
):
    payload = _payload(job_file)
    change(payload["training_data"])
    _write(job_file, payload)

    with pytest.raises(ValueError, match=message):
        ModuleJob.load(job_file)


def test_job_rejects_wrong_training_data_extension(job_file: Path):
    payload = _payload(job_file)
    source = Path(payload["training_data"]["path"])
    wrong = source.with_suffix(".json")
    wrong.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    payload["training_data"]["path"] = str(wrong)
    _write(job_file, payload)

    with pytest.raises(ValueError, match="extension"):
        ModuleJob.load(job_file)


def test_job_requires_reference_when_evaluation_is_enabled(job_file: Path):
    payload = _payload(job_file)
    payload["reference"] = None
    _write(job_file, payload)

    with pytest.raises(ValueError, match="reference"):
        ModuleJob.load(job_file)


def test_job_allows_no_reference_without_evaluation(job_file: Path):
    payload = _payload(job_file)
    payload["stages"] = ["preprocess", "s2", "s1"]
    payload["reference"] = None
    _write(job_file, payload)

    assert ModuleJob.load(job_file).reference is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda reference: reference.update(language="mixed"), "reference.language"),
        (lambda reference: reference.update(text=""), "reference.text"),
        (lambda reference: reference.update(extra=True), "unknown reference field"),
    ],
)
def test_job_strictly_validates_reference(job_file: Path, change, message: str):
    payload = _payload(job_file)
    change(payload["reference"])
    _write(job_file, payload)

    with pytest.raises(ValueError, match=message):
        ModuleJob.load(job_file)


def test_job_accepts_reference_audio_outside_project_as_read_only_input(job_file: Path):
    outside = job_file.parents[3] / "outside.wav"
    outside.write_bytes(b"outside")
    payload = _payload(job_file)
    payload["reference"]["audio"] = str(outside.resolve())
    _write(job_file, payload)

    loaded = ModuleJob.load(job_file)
    assert loaded.reference is not None
    assert loaded.reference.audio == outside.resolve()


def test_job_rejects_symlink_escape(job_file: Path):
    project_root = Path(_payload(job_file)["project_root"])
    outside = project_root.parent / "outside"
    outside.mkdir()
    link = project_root / "linked-output"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    payload = _payload(job_file)
    payload["output_root"] = str(link / "runs")
    _write(job_file, payload)

    with pytest.raises(ValueError, match="output_root"):
        ModuleJob.load(job_file)
