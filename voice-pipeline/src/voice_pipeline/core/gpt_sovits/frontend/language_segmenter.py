"""Local-model language segmentation for ZH/JA/EN mixed text."""

from pathlib import Path

from fast_langdetect import LangDetectConfig, LangDetector


_DETECTED_LANGUAGE_MAP = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "yue": "zh",
    "wuu": "zh",
    "ja": "ja",
    "en": "en",
}


def _script_kind(character: str) -> str | None:
    codepoint = ord(character)
    if "A" <= character <= "Z" or "a" <= character <= "z":
        return "en"
    if (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x323AF
    ):
        return "cjk"
    return None


class LanguageSegmenter:
    def __init__(self, model_dir: str | Path) -> None:
        model_path = Path(model_dir) / "lid.176.bin"
        if not model_path.is_file():
            raise FileNotFoundError(f"fast-langdetect model not found: {model_path}")
        config = LangDetectConfig(custom_model_path=str(model_path), model="full")
        self._detector = LangDetector(config)

    def _detect_language(self, text: str) -> str:
        result = self._detector.detect(text, model="full")
        if not result:
            raise ValueError(f"language detector returned no result for: {text!r}")
        return str(result[0]["lang"])

    def _map_detected_language(self, text: str) -> str:
        detected = self._detect_language(text)
        try:
            return _DETECTED_LANGUAGE_MAP[detected]
        except KeyError as error:
            raise ValueError(f"unsupported detected language: {detected}") from error

    def segment(self, text: str) -> list[tuple[str, str]]:
        if not text.strip():
            raise ValueError("text must not be empty")

        raw_spans: list[tuple[str, str]] = []
        pending = ""
        current_kind: str | None = None
        current_text = ""
        for character in text:
            kind = _script_kind(character)
            if kind is None:
                if current_kind is None:
                    pending += character
                else:
                    current_text += character
                continue
            if current_kind is None:
                current_kind = kind
                current_text = pending + character
                pending = ""
            elif kind == current_kind:
                current_text += character
            else:
                raw_spans.append((current_kind, current_text))
                current_kind = kind
                current_text = character

        if current_kind is None:
            raw_spans.append(("cjk", pending))
        else:
            raw_spans.append((current_kind, current_text + pending))

        spans: list[tuple[str, str]] = []
        for kind, span_text in raw_spans:
            language = "en" if kind == "en" else self._map_detected_language(span_text)
            if spans and spans[-1][0] == language:
                previous_language, previous_text = spans[-1]
                spans[-1] = previous_language, previous_text + span_text
            else:
                spans.append((language, span_text))
        return spans
