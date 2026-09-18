from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from voice_pipeline.module_api.descriptor import build_descriptor
from voice_pipeline.module_api.job import ModuleJob, ModuleReference
from voice_pipeline.pipeline.config import PipelineSpec
from voice_pipeline.training.config import TrainingConfig
from voice_pipeline.training.manifest import read_manifest_records
from voice_pipeline.training.preprocess.config import PreprocessConfig


@dataclass(frozen=True, slots=True)
class MaterializedJob:
    training_config: Path
    pipeline_spec: Path
    run_dir: Path


def materialize_job(job: ModuleJob) -> MaterializedJob:
    module_dir = job.job_dir / "module"
    module_dir.mkdir(parents=True, exist_ok=True)
    training_path = module_dir / "train.yaml"
    pipeline_path = module_dir / "pipeline.yaml"
    values = _parameter_values(job)

    training = {
        "profile": {"name": job.framework},
        "experiment": {"name": job.project_name, "output_root": str(job.output_root)},
        "device": {"device": job.device, "precision": job.precision},
        "dataset": {"manifest": str(job.training_data.path)},
        "objective": {
            "training_languages": _training_languages(job),
            "target_languages": ["zh", "ja", "en"],
            "cross_language_preservation": "strict",
        },
        "preprocess": {"resume": values["preprocess.resume"]},
        "s2": _s2_config(job, values),
        "s1": _s1_config(job, values),
        "evaluation": _evaluation_config(job, values),
    }
    relative_training = training_path.relative_to(job.project_root).as_posix()
    pipeline = {
        "schema_version": 1,
        "config": relative_training,
        "stages": list(job.stages),
    }

    _atomic_yaml(training_path, training)
    _atomic_yaml(pipeline_path, pipeline)
    PipelineSpec.load(pipeline_path, job.project_root)
    PreprocessConfig.from_yaml(training_path, job.project_root)
    if {"s2", "s1", "evaluate"} & set(job.stages):
        TrainingConfig.from_yaml(training_path, job.project_root)

    return MaterializedJob(
        training_config=training_path.resolve(),
        pipeline_spec=pipeline_path.resolve(),
        run_dir=(job.output_root / job.project_name).resolve(),
    )


def _parameter_values(job: ModuleJob) -> dict[str, object]:
    framework = next(
        item for item in build_descriptor()["frameworks"] if item["id"] == job.framework
    )
    values = {field["key"]: field["default"] for field in framework["fields"]}
    values.update(job.parameters)
    return values


def _s2_config(job: ModuleJob, values: dict[str, object]) -> dict[str, object]:
    if "s2" not in job.stages:
        return {"enabled": False}
    return {
        "enabled": True,
        "batch_size": values["s2.batch_size"],
        "target_steps": values["s2.target_steps"],
        "checkpoint_every_steps": 200,
        "learning_rate": values["s2.learning_rate"],
        "text_low_lr_rate": 0.4,
        "freeze_quantizer": True,
        "grad_ckpt": False,
    }


def _s1_config(job: ModuleJob, values: dict[str, object]) -> dict[str, object]:
    if "s1" not in job.stages:
        return {"enabled": False}
    return {
        "enabled": True,
        "batch_size": values["s1.batch_size"],
        "gradient_accumulation": 4,
        "target_optimizer_steps": values["s1.target_optimizer_steps"],
        "checkpoint_every_steps": 100,
    }


def _evaluation_config(job: ModuleJob, values: dict[str, object]) -> dict[str, object]:
    if "evaluate" not in job.stages:
        return {"enabled": False}
    reference = job.reference or _reference_from_training_data(job)
    suites = {
        language: job.project_root / "configs" / "eval" / f"{language}.txt"
        for language in ("zh", "ja", "en", "mixed")
    }
    missing = [path for path in suites.values() if not path.is_file()]
    if missing:
        raise ValueError(f"evaluation suite does not exist: {missing[0]}")
    hf_home = os.environ.get("HF_HOME")
    cache_dir = (
        (Path(hf_home).resolve() / "hub")
        if hf_home
        else job.project_root / "models" / "evaluators"
    )
    return {
        "enabled": True,
        "reference": {
            "audio": str(reference.audio),
            "text": reference.text,
            "language": reference.language,
        },
        "speaker_references": [str(reference.audio)],
        "suites": {language: str(path) for language, path in suites.items()},
        "models": {"cache_dir": str(cache_dir)},
        "pairing": {
            "s2_keep": 2,
            "shortlist_size": values["evaluation.shortlist_size"],
        },
    }


def _reference_from_training_data(job: ModuleJob) -> ModuleReference:
    records = read_manifest_records(job.training_data.path).records
    for record in records:
        item = record.item
        if item.language not in {"zh", "ja", "en"}:
            continue
        if not _reference_audio_usable(item.audio_path):
            continue
        return ModuleReference(item.audio_path, item.text, item.language)
    raise ValueError(
        "evaluation requires at least one decodable 3-10 second zh, ja, or en training sample"
    )


def _reference_audio_usable(path: Path) -> bool:
    try:
        process = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
                "-ar", "32000", "-",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    if process.returncode or len(process.stdout) % 4:
        return False
    samples = array("f")
    samples.frombytes(process.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    return 3 * 32_000 <= len(samples) <= 10 * 32_000 and all(
        math.isfinite(sample) for sample in samples
    )


def _training_languages(job: ModuleJob) -> list[str]:
    records = read_manifest_records(job.training_data.path).records
    present = {record.item.language for record in records}
    languages = [language for language in ("zh", "ja", "en") if language in present]
    if not languages and job.reference is not None:
        languages = [job.reference.language]
    if not languages:
        raise ValueError("dataset.list contains no usable training language")
    return languages


def _atomic_yaml(path: Path, payload: dict[str, object]) -> None:
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["MaterializedJob", "materialize_job"]
