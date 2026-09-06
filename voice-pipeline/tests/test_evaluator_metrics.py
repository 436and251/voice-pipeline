import pytest

from voice_pipeline.evaluation.base import MetricResult, Transcript
from voice_pipeline.evaluation.language_consistency import language_consistency
from voice_pipeline.evaluation.pronunciation import (
    character_error_rate,
    evaluate_pronunciation,
    word_error_rate,
)


def test_chinese_cer_is_hand_calculated() -> None:
    assert character_error_rate("你好世界", "你好世") == pytest.approx(0.25)


def test_japanese_cer_normalizes_width_space_and_punctuation() -> None:
    assert character_error_rate("ＡＩ を始めます。", "AIを始めます") == 0.0


def test_english_wer_is_hand_calculated() -> None:
    assert word_error_rate("the quick brown fox", "the quick fox") == pytest.approx(0.25)


def test_english_wer_normalizes_case_and_punctuation() -> None:
    assert word_error_rate("Hello, WORLD!", "hello world") == 0.0


def test_error_rate_rejects_empty_normalized_target() -> None:
    with pytest.raises(ValueError, match="target text"):
        character_error_rate("...", "anything")
    with pytest.raises(ValueError, match="target text"):
        word_error_rate("...", "anything")


def test_pronunciation_uses_cer_for_zh_and_ja_and_wer_for_en() -> None:
    zh = evaluate_pronunciation("你好世界", Transcript("你好世", "zh", 0.9), "zh")
    ja = evaluate_pronunciation("今日は晴れ", Transcript("今日は晴", "ja", 0.9), "ja")
    en = evaluate_pronunciation("one two three", Transcript("one three", "en", 0.9), "en")

    assert (zh.name, zh.value, zh.details["unit"]) == ("pronunciation", pytest.approx(0.25), "cer")
    assert (ja.name, ja.value, ja.details["unit"]) == ("pronunciation", pytest.approx(0.2), "cer")
    assert (en.name, en.value, en.details["unit"]) == ("pronunciation", pytest.approx(1 / 3), "wer")


def test_single_language_consistency_uses_detected_language() -> None:
    matching = language_consistency("hello", "hello", "en", detected_language="en", probability=0.8)
    drifting = language_consistency("hello", "hello", "en", detected_language="ja", probability=0.9)

    assert matching == MetricResult("language_consistency", 0.8, {"detected_language": "en"}, True)
    assert drifting.value == 0.0
    assert drifting.details["detected_language"] == "ja"


def test_mixed_language_does_not_require_one_detected_language() -> None:
    result = language_consistency(
        "你好 hello",
        "你好 hello",
        "mixed",
        detected_language="zh",
        probability=0.99,
    )

    assert result.available is True
    assert result.value == pytest.approx(1.0)
    assert result.details["required_scripts"] == ["han", "latin"]


def test_mixed_language_detects_dropped_script() -> None:
    result = language_consistency(
        "请打开 settings",
        "请打开",
        "mixed",
        detected_language="zh",
        probability=0.99,
    )

    assert result.value == pytest.approx(0.5)
    assert result.details["present_scripts"] == ["han"]
