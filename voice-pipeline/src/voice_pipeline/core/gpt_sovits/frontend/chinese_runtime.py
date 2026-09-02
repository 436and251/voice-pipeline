"""Local-only production wiring for the GPT-SoVITS Chinese frontend."""

from pathlib import Path

import jieba_fast.posseg as psg
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from .chinese import g2p, replace_consecutive_punctuation, replace_punctuation
from .g2pw import G2PWPinyin, correct_pronunciation
from .g2pw.onnx_api import validate_model_dir
from .tone_sandhi import ToneSandhi
from .zh_normalization import TextNormalizer


class _BatchG2PW:
    def __init__(self, g2pw: G2PWPinyin) -> None:
        self._g2pw = g2pw

    def batch(self, texts: list[str]) -> list[list[str]]:
        return self._g2pw._g2pw(texts)


def _split_pinyin(token: str) -> tuple[str, str]:
    if token and token[0].isalpha():
        return (
            to_initials(token, strict=True),
            to_finals_tone3(token, strict=True, neutral_tone_with_five=True),
        )
    return token, token


class ChineseFrontend:
    """Normalize Chinese text and produce OpenCPOP phones using local assets."""

    def __init__(self, g2pw_model_path: str | Path, bert_path: str | Path) -> None:
        model_dir = validate_model_dir(str(g2pw_model_path))
        bert_dir = Path(bert_path)
        if not bert_dir.is_dir():
            raise FileNotFoundError(f"Chinese BERT directory not found: {bert_dir}")

        g2pw = G2PWPinyin(
            model_dir=model_dir,
            model_source=str(bert_dir),
            v_to_u=False,
            neutral_tone_with_five=True,
        )
        self._normalizer = TextNormalizer()
        self._tone_sandhi = ToneSandhi()
        self._pinyin_provider = _BatchG2PW(g2pw)

    def normalize(self, text: str) -> str:
        normalized = "".join(
            replace_punctuation(sentence) for sentence in self._normalizer.normalize(text)
        )
        return replace_consecutive_punctuation(normalized)

    def process(self, text: str) -> tuple[str, list[str], list[int]]:
        normalized = self.normalize(text)
        phones, word2ph = g2p(
            normalized,
            segmenter=psg.lcut,
            tone_modifier=self._tone_sandhi,
            pinyin_provider=self._pinyin_provider,
            pinyin_converter=_split_pinyin,
            pronunciation_corrector=correct_pronunciation,
        )
        return normalized, phones, word2ph
