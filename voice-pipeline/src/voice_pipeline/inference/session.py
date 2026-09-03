from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import random
import threading

import numpy as np
import torch

from voice_pipeline.common.model_bundle import ModelBundle
from voice_pipeline.core.gpt_sovits.compatibility.s1_checkpoint import load_s1_checkpoint
from voice_pipeline.core.gpt_sovits.compatibility.s2_checkpoint import load_s2_generator
from voice_pipeline.core.gpt_sovits.features.cnhubert import CNHubertExtractor
from voice_pipeline.core.gpt_sovits.features.speaker import SpeakerEncoder
from voice_pipeline.core.gpt_sovits.frontend.multilingual import MultilingualFrontend
from voice_pipeline.profiles.registry import ProfileRegistry

from .acoustic import decode_waveform
from .reference import ReferenceCondition, build_reference_condition
from .result import InferenceIdentity, InferenceResult
from .semantic import generate_semantic


class InferenceSession:
    def __init__(
        self,
        *,
        s1,
        s2,
        frontend,
        reference: ReferenceCondition,
        identity: InferenceIdentity,
        device: torch.device,
        dtype: torch.dtype,
        sample_rate: int,
    ) -> None:
        self.s1 = s1
        self.s2 = s2
        self.frontend = frontend
        self.reference = reference
        self.identity = identity
        self.device = device
        self.dtype = dtype
        self.sample_rate = sample_rate
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        bundle_path: str | Path,
        device: str | torch.device = "cpu",
        *,
        reference_audio: str | Path | None = None,
        reference_text: str | None = None,
        reference_language: str | None = None,
    ) -> "InferenceSession":
        _validate_reference_override(reference_audio, reference_text, reference_language)
        bundle = ModelBundle.load(Path(bundle_path))
        profile = ProfileRegistry.get(bundle.profile)
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        project_root = Path.cwd().resolve()

        s1 = load_s1_checkpoint(bundle.root / bundle.weights["s1"], device).eval().to(dtype=dtype)
        s2 = load_s2_generator(bundle.root / bundle.weights["s2"], device).eval().to(dtype=dtype)
        frontend = MultilingualFrontend(
            project_root / profile.bert_relative_path,
            project_root / profile.g2pw_relative_path,
            project_root / profile.nltk_data_relative_path,
            project_root / profile.langdetect_relative_path,
            device,
        )
        if dtype == torch.float16:
            frontend.bert.model.half()
        hubert = CNHubertExtractor(
            project_root / profile.hubert_relative_path,
            device,
            precision="fp16" if dtype == torch.float16 else "fp32",
        )
        speaker = SpeakerEncoder(
            project_root / profile.speaker_relative_path,
            device,
            precision="fp16" if dtype == torch.float16 else "fp32",
        )
        if reference_audio is None:
            selected_audio = (bundle.root / bundle.reference.audio).resolve()
            selected_text = bundle.reference.text
            selected_language = bundle.reference.language
        else:
            selected_audio = Path(reference_audio).resolve()
            if not selected_audio.is_file():
                raise FileNotFoundError(selected_audio)
            selected_text = reference_text
            selected_language = reference_language
        reference = build_reference_condition(
            selected_audio,
            selected_text,
            selected_language,
            frontend=frontend,
            hubert=hubert,
            speaker=speaker,
            s2=s2,
            device=device,
            dtype=dtype,
        )
        identity = InferenceIdentity(
            model_name=bundle.metadata["model_name"],
            s1_sha256=bundle.metadata["checkpoints"]["s1"]["exported_sha256"],
            s2_sha256=bundle.metadata["checkpoints"]["s2"]["exported_sha256"],
            reference_sha256=_sha256(selected_audio),
            reference_text=selected_text,
            reference_language=selected_language,
        )
        return cls(
            s1=s1,
            s2=s2,
            frontend=frontend,
            reference=reference,
            identity=identity,
            device=device,
            dtype=dtype,
            sample_rate=profile.sample_rate,
        )

    def synthesize(
        self,
        text: str,
        language: str,
        *,
        seed: int = 0,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
    ) -> InferenceResult:
        _validate_options(
            text, language, seed, top_k, top_p, temperature, repetition_penalty, noise_scale, speed
        )
        with self._lock, _fixed_seed(seed, self.device):
            semantic = generate_semantic(
                text,
                language,
                frontend=self.frontend,
                s1=self.s1,
                reference=self.reference,
                device=self.device,
                dtype=self.dtype,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                early_stop_num=57 * 50,
            )
            waveform = decode_waveform(
                semantic,
                s2=self.s2,
                reference=self.reference,
                noise_scale=noise_scale,
                speed=speed,
            )
        return InferenceResult(waveform, self.sample_rate, seed)


def _validate_reference_override(reference_audio, reference_text, reference_language) -> None:
    if reference_audio is None:
        if reference_text is not None or reference_language is not None:
            raise ValueError("reference_text/reference_language cannot be used without reference_audio")
        return
    if reference_language is None:
        raise ValueError("reference_audio and reference_language must be provided together")
    if reference_language not in {"zh", "ja", "en"}:
        raise ValueError("reference_language must be zh, ja, or en")
    if reference_text is not None and (not isinstance(reference_text, str) or not reference_text.strip()):
        raise ValueError("reference_text must be a non-empty string when present")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_options(text, language, seed, top_k, top_p, temperature, repetition_penalty, noise_scale, speed):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    if language not in {"zh", "ja", "en", "mixed"}:
        raise ValueError("language must be zh, ja, en, or mixed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be a nonnegative 64-bit integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    _positive_float("top_p", top_p, upper=1.0)
    _positive_float("temperature", temperature)
    _positive_float("repetition_penalty", repetition_penalty)
    _positive_float("speed", speed)
    if isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float)) or not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be a finite nonnegative number")


def _positive_float(name: str, value, *, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}")


@contextmanager
def _fixed_seed(seed: int, device: torch.device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


__all__ = ["InferenceSession"]
