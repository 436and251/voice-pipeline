from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice_pipeline.core.gpt_sovits.frontend.english import EnglishFrontend


def _nltk_data_dir() -> Path:
    value = os.environ.get("VOICE_PIPELINE_TEST_NLTK_DATA")
    if not value:
        pytest.skip("set VOICE_PIPELINE_TEST_NLTK_DATA for real English frontend tests")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"missing English NLTK data directory: {path}")
    return path


def test_english_runtime_uses_official_cmu_path() -> None:
    normalized, phones, word2ph = EnglishFrontend(_nltk_data_dir()).process("Hello world.")

    assert normalized == "Hello world."
    assert phones == ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D", "."]
    assert word2ph is None


def test_english_runtime_requires_current_nltk_tagger(tmp_path: Path) -> None:
    with pytest.raises(LookupError, match="averaged_perceptron_tagger_eng"):
        EnglishFrontend(tmp_path).process("Hello.")


def test_english_runtime_preserves_pinned_number_normalization() -> None:
    normalized, phones, _ = EnglishFrontend(_nltk_data_dir()).process("I have 2 cats.")

    assert normalized == "I have two cats."
    assert "T" in phones
    assert "UW1" in phones
