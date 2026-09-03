from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from voice_pipeline.inference.long_text import synthesize_text
from voice_pipeline.inference.result import InferenceResult


class FakeSession:
    def __init__(self, *, sample_rate: int = 32_000, waveform: np.ndarray | None = None):
        self.sample_rate = sample_rate
        self.waveform = np.array([0.25, -0.25], dtype=np.float32) if waveform is None else waveform
        self.calls = []

    def synthesize(self, text, language, **options):
        self.calls.append(SimpleNamespace(text=text, language=language, **options))
        return InferenceResult(self.waveform.copy(), self.sample_rate, options["seed"])


def test_long_text_uses_derived_seeds_and_default_ten_ms_pause():
    session = FakeSession()

    result = synthesize_text(session, "aa。bb。cc。", "zh", max_chars=3, seed=8)

    assert [call.text for call in session.calls] == ["aa。", "bb。", "cc。"]
    assert [call.seed for call in session.calls] == [8, 9, 10]
    assert all(call.language == "zh" and call.top_k == 5 for call in session.calls)
    assert result.sample_rate == 32_000 and result.seed == 8
    assert result.waveform.dtype == np.float32
    assert result.waveform.shape == (6 + 2 * 320,)
    assert np.count_nonzero(result.waveform) == 6


@pytest.mark.parametrize("pause_ms, silence_samples", [(0, 0), (25, 800)])
def test_long_text_pause_is_configurable(pause_ms: int, silence_samples: int):
    result = synthesize_text(
        FakeSession(),
        "aa。bb。",
        "zh",
        max_chars=3,
        pause_ms=pause_ms,
    )
    assert result.waveform.shape == (4 + silence_samples,)


@pytest.mark.parametrize("pause_ms", [-1, True, 1.5])
def test_long_text_rejects_invalid_pause_before_session_access(pause_ms):
    with pytest.raises(ValueError, match="pause_ms"):
        synthesize_text(object(), "text", "en", pause_ms=pause_ms)


def test_long_text_rejects_inconsistent_chunk_results():
    class ChangingSession(FakeSession):
        def synthesize(self, text, language, **options):
            result = super().synthesize(text, language, **options)
            if len(self.calls) == 2:
                return InferenceResult(result.waveform, 24_000, result.seed)
            return result

    with pytest.raises(RuntimeError, match="sample rate"):
        synthesize_text(ChangingSession(), "aa。bb。", "zh", max_chars=3)

    with pytest.raises(RuntimeError, match="mono"):
        synthesize_text(FakeSession(waveform=np.zeros((1, 2), dtype=np.float32)), "text", "en")
