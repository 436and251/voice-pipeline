from __future__ import annotations

import re
import unicodedata

from .base import MetricResult, Transcript


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def character_error_rate(target: str, transcript: str) -> float:
    expected = _characters(target)
    actual = _characters(transcript)
    if not expected:
        raise ValueError("target text is empty after normalization")
    return _edit_distance(expected, actual) / len(expected)


def word_error_rate(target: str, transcript: str) -> float:
    expected = _words(target)
    actual = _words(transcript)
    if not expected:
        raise ValueError("target text is empty after normalization")
    return _edit_distance(expected, actual) / len(expected)


def evaluate_pronunciation(target: str, transcript: Transcript, language: str) -> MetricResult:
    if language not in {"zh", "ja", "en", "mixed"}:
        raise ValueError("language must be zh, ja, en, or mixed")
    if language == "en":
        value = word_error_rate(target, transcript.text)
        unit = "wer"
        normalized_target = " ".join(_words(target))
        normalized_transcript = " ".join(_words(transcript.text))
    else:
        value = character_error_rate(target, transcript.text)
        unit = "cer"
        normalized_target = "".join(_characters(target))
        normalized_transcript = "".join(_characters(transcript.text))
    return MetricResult(
        "pronunciation",
        value,
        {
            "unit": unit,
            "target": target,
            "transcript": transcript.text,
            "normalized_target": normalized_target,
            "normalized_transcript": normalized_transcript,
        },
        True,
    )


def _characters(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(character for character in normalized if unicodedata.category(character)[0] in {"L", "N"})


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def _edit_distance(expected: tuple[str, ...], actual: tuple[str, ...]) -> int:
    previous = list(range(len(actual) + 1))
    for row, expected_item in enumerate(expected, 1):
        current = [row]
        for column, actual_item in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_item != actual_item),
                )
            )
        previous = current
    return previous[-1]


__all__ = ["character_error_rate", "evaluate_pronunciation", "word_error_rate"]
