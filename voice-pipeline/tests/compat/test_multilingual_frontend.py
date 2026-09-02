from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def _asset(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"set {env_name} for real multilingual frontend tests")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"missing multilingual frontend asset: {path}")
    return path


@pytest.fixture(scope="module")
def frontend():
    from voice_pipeline.core.gpt_sovits.frontend.multilingual import MultilingualFrontend

    return MultilingualFrontend(
        bert_path=_asset("VOICE_PIPELINE_TEST_BERT_DIR"),
        g2pw_path=_asset("VOICE_PIPELINE_TEST_G2PW_DIR"),
        nltk_data_path=_asset("VOICE_PIPELINE_TEST_NLTK_DATA"),
        langdetect_path=_asset("VOICE_PIPELINE_TEST_LANGDETECT_DIR"),
    )


def test_explicit_languages_use_their_own_frontends(frontend) -> None:
    zh = frontend.process("重庆。", "zh")
    ja = frontend.process("こんにちは。", "ja")
    en = frontend.process("Hello world.", "en")

    assert zh.phones[:2] == ["ch", "ong2"]
    assert ja.phones[:2] == ["k", "o"]
    assert en.phones[:2] == ["HH", "AH0"]
    assert zh.word2ph is not None
    assert ja.word2ph is None and en.word2ph is None
    assert torch.count_nonzero(zh.bert_features) > 0
    assert torch.count_nonzero(ja.bert_features) == 0
    assert torch.count_nonzero(en.bert_features) == 0


def test_mixed_result_concatenates_aligned_language_spans(frontend) -> None:
    result = frontend.process("你好。Hello world.", "mixed")

    assert result.word2ph is None
    assert result.bert_features.shape == (1024, len(result.phone_ids))
    assert "HH" in result.phones
    assert torch.count_nonzero(result.bert_features) > 0


def test_frontend_rejects_empty_text_and_unsupported_language(frontend) -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        frontend.process("", "en")
    with pytest.raises(ValueError, match="unsupported language: ko"):
        frontend.process("hello", "ko")


def test_language_segmenter_rejects_missing_model(tmp_path: Path) -> None:
    from voice_pipeline.core.gpt_sovits.frontend.language_segmenter import LanguageSegmenter

    with pytest.raises(FileNotFoundError, match="lid.176.bin"):
        LanguageSegmenter(tmp_path)


def test_language_segmenter_rejects_unknown_detector_output(monkeypatch) -> None:
    from voice_pipeline.core.gpt_sovits.frontend.language_segmenter import LanguageSegmenter

    segmenter = LanguageSegmenter(_asset("VOICE_PIPELINE_TEST_LANGDETECT_DIR"))
    monkeypatch.setattr(segmenter, "_detect_language", lambda text: "ko")

    with pytest.raises(ValueError, match="unsupported detected language: ko"):
        segmenter.segment("你好。")


def test_language_segmenter_preserves_order_and_attaches_punctuation() -> None:
    from voice_pipeline.core.gpt_sovits.frontend.language_segmenter import LanguageSegmenter

    segmenter = LanguageSegmenter(_asset("VOICE_PIPELINE_TEST_LANGDETECT_DIR"))

    assert segmenter.segment("你好。Hello world.こんにちは。") == [
        ("zh", "你好。"),
        ("en", "Hello world."),
        ("ja", "こんにちは。"),
    ]
