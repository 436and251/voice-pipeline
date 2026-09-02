from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch
import torchaudio

from voice_pipeline.training.manifest import ManifestRecord

from .artifacts import atomic_torch_save, sha256_file
from .base import SampleFailure, StageContext, StageSampleResult


class SpeakerExtractor(Protocol):
    def extract(self, wav_32k: torch.Tensor) -> torch.Tensor: ...


class SVStage:
    name = "sv"
    dependencies = {"wav32k"}

    def __init__(self, encoder: SpeakerExtractor):
        self.encoder = encoder

    def _wav_path(self, record: ManifestRecord, context: StageContext) -> Path:
        return context.preprocess_dir / "wav32k" / f"{record.sample_id}.wav"

    def signature(self, record: ManifestRecord, context: StageContext) -> str:
        payload = {
            "stage": self.name,
            "version": 1,
            "wav32_sha256": sha256_file(self._wav_path(record, context)),
            "model_sha256": context.asset_digests.get("speaker"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def run(self, record: ManifestRecord, context: StageContext) -> StageSampleResult:
        try:
            waveform, sample_rate = torchaudio.load(self._wav_path(record, context))
        except (OSError, RuntimeError) as error:
            raise SampleFailure(self.name, "invalid_wav32k", str(error)) from error
        if sample_rate != 32_000 or waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] < 1:
            raise SampleFailure(
                self.name,
                "invalid_wav32k",
                f"expected mono 32 kHz WAV, got shape {list(waveform.shape)} at {sample_rate} Hz",
            )
        try:
            embedding = self.encoder.extract(waveform)
        except RuntimeError as error:
            raise SampleFailure(self.name, "feature_extraction", str(error)) from error
        if embedding.shape != (1, 20_480):
            raise SampleFailure(self.name, "invalid_shape", f"unexpected SV shape: {list(embedding.shape)}")
        if not torch.isfinite(embedding).all():
            raise SampleFailure(self.name, "nonfinite", "speaker embedding contains non-finite values")

        embedding = embedding.detach().cpu().contiguous()
        output_path = context.preprocess_dir / self.name / f"{record.sample_id}.pt"
        atomic_torch_save(output_path, embedding)
        metadata = {"shape": list(embedding.shape), "dtype": str(embedding.dtype)}
        return StageSampleResult(record.sample_id, [output_path], metadata)

    def validate_cached(self, record, entry, context) -> bool:
        output_paths = entry.get("output_paths")
        metadata = entry.get("metadata")
        if not isinstance(output_paths, list) or len(output_paths) != 1 or not isinstance(metadata, dict):
            return False
        try:
            embedding = torch.load(Path(output_paths[0]), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.shape == (1, 20_480)
            and torch.isfinite(embedding).all().item()
            and metadata.get("shape") == [1, 20_480]
            and metadata.get("dtype") == str(embedding.dtype)
        )
