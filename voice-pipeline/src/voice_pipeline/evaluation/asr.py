from __future__ import annotations

from pathlib import Path

from .base import Transcript


class WhisperEvaluator:
    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def from_model(cls, model) -> "WhisperEvaluator":
        return cls(model)

    @classmethod
    def load(
        cls,
        model_id_or_path: str,
        *,
        cache_dir: Path,
        device: str,
        precision: str,
    ) -> "WhisperEvaluator":
        model_path = Path(model_id_or_path)
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download

            model_path = Path(snapshot_download(model_id_or_path, cache_dir=cache_dir))
        from faster_whisper import WhisperModel

        device_type, _, raw_index = device.partition(":")
        device_index = int(raw_index) if raw_index else 0
        compute_type = "float16" if device_type == "cuda" and precision == "fp16" else "float32"
        model = WhisperModel(
            str(model_path),
            device=device_type,
            device_index=device_index,
            compute_type=compute_type,
        )
        return cls(model)

    def transcribe(self, audio: Path, language: str | None) -> Transcript:
        audio = Path(audio)
        if not audio.is_file():
            raise ValueError(f"ASR audio does not exist: {audio}")
        if language not in {None, "zh", "ja", "en"}:
            raise ValueError("ASR language must be zh, ja, en, or None")
        segments, info = self._model.transcribe(
            str(audio),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700},
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcript(
            text,
            getattr(info, "language", None),
            getattr(info, "language_probability", None),
        )


__all__ = ["WhisperEvaluator"]
