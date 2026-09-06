from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from voice_pipeline.inference.wav import read_wav


class WavLMSpeakerEvaluator:
    def __init__(self, feature_extractor, model, device: torch.device) -> None:
        self._feature_extractor = feature_extractor
        self._model = model
        self._device = device

    @classmethod
    def from_components(cls, feature_extractor, model, *, device: str) -> "WavLMSpeakerEvaluator":
        return cls(feature_extractor, model, torch.device(device))

    @classmethod
    def load(
        cls,
        model_id_or_path: str,
        *,
        cache_dir: Path,
        device: str,
    ) -> "WavLMSpeakerEvaluator":
        from transformers import AutoFeatureExtractor, WavLMForXVector

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id_or_path, cache_dir=cache_dir)
        model = WavLMForXVector.from_pretrained(model_id_or_path, cache_dir=cache_dir)
        selected_device = torch.device(device)
        model.to(selected_device).eval()
        return cls(feature_extractor, model, selected_device)

    def embedding(self, audio: Path) -> np.ndarray:
        sample_rate, waveform = read_wav(audio)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("speaker audio must be non-empty and finite")
        if sample_rate != 16000:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        inputs = self._feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with torch.inference_mode():
            embedding = self._model(**inputs).embeddings[0].detach().float().cpu().numpy()
        return _normalize(embedding)

    def centroid(self, references: tuple[Path, ...]) -> np.ndarray:
        if not references:
            raise ValueError("at least one speaker reference is required")
        return _normalize(np.mean([self.embedding(path) for path in references], axis=0))

    def similarity(self, audio: Path, centroid: np.ndarray) -> float:
        return float(self.embedding(audio) @ _normalize(centroid))


def _normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("speaker embedding must be non-zero and finite")
    return vector / norm


__all__ = ["WavLMSpeakerEvaluator"]
