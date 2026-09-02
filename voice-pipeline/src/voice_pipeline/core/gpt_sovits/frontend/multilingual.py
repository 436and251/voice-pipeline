"""Unified GPT-SoVITS ZH/JA/EN training frontend."""

from pathlib import Path

import torch

from .bert import BertAligner
from .chinese_runtime import ChineseFrontend
from .contract import FrontendResult
from .english import EnglishFrontend
from .japanese_runtime import JapaneseFrontend
from .language_segmenter import LanguageSegmenter
from .symbols import phone_ids


class MultilingualFrontend:
    def __init__(
        self,
        bert_path: str | Path,
        g2pw_path: str | Path,
        nltk_data_path: str | Path,
        langdetect_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.chinese = ChineseFrontend(g2pw_path, bert_path)
        self.japanese = JapaneseFrontend()
        self.english = EnglishFrontend(nltk_data_path)
        self.segmenter = LanguageSegmenter(langdetect_path)
        self.bert = BertAligner(bert_path, device)

    def _process_span(self, text: str, language: str) -> FrontendResult:
        if language == "zh":
            normalized, phones, word2ph = self.chinese.process(text)
            assert word2ph is not None
            bert_features = self.bert.extract(normalized, word2ph)
        elif language == "ja":
            normalized, phones, word2ph = self.japanese.process(text)
            bert_features = self._zero_bert(len(phones))
        elif language == "en":
            normalized, phones, word2ph = self.english.process(text)
            bert_features = self._zero_bert(len(phones))
        else:
            raise ValueError(f"unsupported language: {language}")
        return FrontendResult(normalized, phones, phone_ids(phones), word2ph, bert_features)

    def _zero_bert(self, phone_count: int) -> torch.Tensor:
        parameter = next(self.bert.model.parameters())
        return torch.zeros((1024, phone_count), dtype=parameter.dtype, device=self.bert.device)

    def process(self, text: str, language: str) -> FrontendResult:
        if language not in {"zh", "ja", "en", "mixed"}:
            raise ValueError(f"unsupported language: {language}")
        if not text.strip():
            raise ValueError("text must not be empty")

        spans = [(language, text)] if language != "mixed" else self.segmenter.segment(text)
        results = [self._process_span(span_text, span_language) for span_language, span_text in spans]
        phones = [phone for result in results for phone in result.phones]
        return FrontendResult(
            normalized_text="".join(result.normalized_text for result in results),
            phones=phones,
            phone_ids=phone_ids(phones),
            word2ph=results[0].word2ph if language == "zh" else None,
            bert_features=torch.cat([result.bert_features for result in results], dim=1),
        )
