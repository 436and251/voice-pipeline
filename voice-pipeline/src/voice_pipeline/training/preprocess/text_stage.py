from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch

from voice_pipeline.core.gpt_sovits.frontend.contract import FrontendResult
from voice_pipeline.training.manifest import ALLOWED_LANGUAGES, ManifestRecord

from .artifacts import atomic_torch_save, atomic_write_text
from .base import SampleFailure, StageContext, StageSampleResult


class Frontend(Protocol):
    def process(self, text: str, language: str) -> FrontendResult: ...


class TextStage:
    name = "text"
    dependencies: set[str] = set()

    def __init__(self, frontend: Frontend):
        self.frontend = frontend

    def signature(self, record: ManifestRecord, context: StageContext) -> str:
        payload = {
            "stage": self.name,
            "version": 1,
            "profile": context.profile.name,
            "language": record.item.language,
            "text": record.item.text,
            "assets": {
                name: context.asset_digests.get(name)
                for name in ("bert", "g2pw", "nltk", "langdetect")
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def run(self, record: ManifestRecord, context: StageContext) -> StageSampleResult:
        language = record.item.language
        if language not in ALLOWED_LANGUAGES:
            raise SampleFailure(self.name, "unsupported_language", f"unsupported language: {language}")
        try:
            frontend_result = self.frontend.process(record.item.text, language)
        except ValueError as error:
            raise SampleFailure(self.name, "frontend_error", str(error)) from error
        if not frontend_result.phone_ids:
            raise SampleFailure(self.name, "empty_phones", "frontend produced no phones")
        if frontend_result.bert_features.shape != (1024, len(frontend_result.phone_ids)):
            raise SampleFailure(self.name, "bert_alignment", "BERT columns do not match phone IDs")

        output_dir = context.preprocess_dir / self.name
        metadata_path = output_dir / f"{record.sample_id}.json"
        bert_path = output_dir / f"{record.sample_id}.bert.pt"
        bert = frontend_result.bert_features.detach().cpu().contiguous()
        metadata = {
            "sample_id": record.sample_id,
            "language": language,
            "normalized_text": frontend_result.normalized_text,
            "phones": frontend_result.phones,
            "phone_ids": frontend_result.phone_ids,
            "word2ph": frontend_result.word2ph,
            "bert_shape": list(bert.shape),
            "bert_dtype": str(bert.dtype),
        }
        atomic_write_text(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        atomic_torch_save(bert_path, bert)
        return StageSampleResult(
            record.sample_id,
            [metadata_path, bert_path],
            {"bert_shape": list(bert.shape), "bert_dtype": str(bert.dtype)},
        )

    def validate_cached(
        self,
        record: ManifestRecord,
        entry: dict[str, object],
        context: StageContext,
    ) -> bool:
        output_paths = entry.get("output_paths")
        if not isinstance(output_paths, list) or len(output_paths) != 2:
            return False
        metadata_path, bert_path = (Path(value) for value in output_paths)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            bert = torch.load(bert_path, map_location="cpu", weights_only=True)
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
            return False
        return (
            metadata.get("sample_id") == record.sample_id
            and metadata.get("language") == record.item.language
            and metadata.get("phone_ids")
            and metadata.get("bert_shape") == [1024, len(metadata["phone_ids"])]
            and isinstance(bert, torch.Tensor)
            and list(bert.shape) == metadata["bert_shape"]
            and str(bert.dtype) == metadata.get("bert_dtype")
        )
