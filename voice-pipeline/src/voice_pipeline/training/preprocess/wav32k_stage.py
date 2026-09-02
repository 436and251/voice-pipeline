from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.io import wavfile

from voice_pipeline.core.gpt_sovits.features.audio import InvalidSampleAudio, prepare_audio_from_source
from voice_pipeline.training.manifest import ManifestRecord

from .artifacts import sha256_file
from .base import SampleFailure, StageContext, StageSampleResult


class Wav32kStage:
    name = "wav32k"
    dependencies: set[str] = set()

    def signature(self, record: ManifestRecord, context: StageContext) -> str:
        payload = {
            "stage": self.name,
            "version": 1,
            "source_sha256": sha256_file(record.item.audio_path),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def run(self, record: ManifestRecord, context: StageContext) -> StageSampleResult:
        try:
            prepared = prepare_audio_from_source(record.item.audio_path)
        except (InvalidSampleAudio, FileNotFoundError, OSError, RuntimeError) as error:
            raise SampleFailure(self.name, "invalid_audio", str(error)) from error

        output_path = context.preprocess_dir / self.name / f"{record.sample_id}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            wavfile.write(temporary, 32_000, prepared.wav32_int16)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return StageSampleResult(
            record.sample_id,
            [output_path],
            {
                "sample_rate": 32_000,
                "channels": 1,
                "dtype": str(prepared.wav32_int16.dtype),
                "num_samples": int(prepared.wav32_int16.size),
            },
        )

    def validate_cached(
        self,
        record: ManifestRecord,
        entry: dict[str, object],
        context: StageContext,
    ) -> bool:
        output_paths = entry.get("output_paths")
        if not isinstance(output_paths, list) or len(output_paths) != 1:
            return False
        try:
            sample_rate, waveform = wavfile.read(Path(output_paths[0]))
        except (OSError, ValueError):
            return False
        metadata = entry.get("metadata")
        return (
            isinstance(metadata, dict)
            and sample_rate == 32_000
            and waveform.ndim == 1
            and waveform.dtype == np.int16
            and waveform.size > 0
            and metadata.get("num_samples") == waveform.size
        )
