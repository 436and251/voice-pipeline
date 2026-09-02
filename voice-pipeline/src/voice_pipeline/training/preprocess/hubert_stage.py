from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch
from torchaudio.functional import resample

from voice_pipeline.core.gpt_sovits.features.audio import InvalidSampleAudio, prepare_audio_from_source
from voice_pipeline.training.manifest import ManifestRecord

from .artifacts import atomic_torch_save, sha256_file
from .base import SampleFailure, StageContext, StageSampleResult


class HubertExtractor(Protocol):
    def extract(self, wav_16k: torch.Tensor) -> torch.Tensor: ...

    def to_float(self) -> None: ...


class HubertStage:
    name = "hubert"
    dependencies = {"wav32k"}

    def __init__(self, extractor: HubertExtractor, precision: str):
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported HuBERT precision: {precision}")
        self.extractor = extractor
        self.precision = precision

    def signature(self, record: ManifestRecord, context: StageContext) -> str:
        payload = {
            "stage": self.name,
            "version": 1,
            "source_sha256": sha256_file(record.item.audio_path),
            "model_sha256": context.asset_digests.get("hubert"),
            "precision": self.precision,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def run(self, record: ManifestRecord, context: StageContext) -> StageSampleResult:
        try:
            prepared = prepare_audio_from_source(record.item.audio_path)
            wav_16k = resample(prepared.hubert_source_32k, 32_000, 16_000)
            content = self.extractor.extract(wav_16k)
        except (InvalidSampleAudio, FileNotFoundError, OSError, RuntimeError) as error:
            raise SampleFailure(self.name, "feature_extraction", str(error)) from error

        if self.precision == "fp16" and not torch.isfinite(content).all():
            self.extractor.to_float()
            try:
                content = self.extractor.extract(wav_16k.float())
            except RuntimeError as error:
                raise SampleFailure(self.name, "feature_extraction", str(error)) from error
        if content.ndim != 3 or content.shape[0] != 1 or content.shape[1] != 768 or content.shape[2] < 1:
            raise SampleFailure(self.name, "invalid_shape", f"unexpected HuBERT shape: {list(content.shape)}")
        if not torch.isfinite(content).all():
            raise SampleFailure(self.name, "nonfinite", "HuBERT output contains non-finite values")

        content = content.detach().cpu().contiguous()
        output_path = context.preprocess_dir / self.name / f"{record.sample_id}.pt"
        atomic_torch_save(output_path, content)
        metadata = {"shape": list(content.shape), "dtype": str(content.dtype)}
        return StageSampleResult(record.sample_id, [output_path], metadata)

    def validate_cached(self, record, entry, context) -> bool:
        return _validate_tensor_entry(entry, [1, 768])


def _validate_tensor_entry(entry: dict[str, object], shape_prefix: list[int]) -> bool:
    output_paths = entry.get("output_paths")
    metadata = entry.get("metadata")
    if not isinstance(output_paths, list) or len(output_paths) != 1 or not isinstance(metadata, dict):
        return False
    try:
        tensor = torch.load(Path(output_paths[0]), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == len(shape_prefix) + 1
        and list(tensor.shape[: len(shape_prefix)]) == shape_prefix
        and tensor.shape[-1] > 0
        and torch.isfinite(tensor).all().item()
        and metadata.get("shape") == list(tensor.shape)
        and metadata.get("dtype") == str(tensor.dtype)
    )
