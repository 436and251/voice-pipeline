import json
from pathlib import Path

import pytest
import torch

from voice_pipeline.core.gpt_sovits.frontend.contract import FrontendResult
from voice_pipeline.profiles.v2proplus import V2PROPLUS
from voice_pipeline.training.experiment import Experiment
from voice_pipeline.training.manifest import ManifestItem, ManifestRecord
from voice_pipeline.training.preprocess.base import SampleFailure, StageContext
from voice_pipeline.training.preprocess.text_stage import TextStage


class FakeFrontend:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def process(self, text, language):
        self.calls.append((text, language))
        return self.result


def record(language="ja", text="こんにちは。"):
    return ManifestRecord(1, "sample", ManifestItem(Path("voice.wav"), "speaker", language, text))


def context(tmp_path):
    return StageContext(Experiment.create("run", tmp_path), V2PROPLUS, {}, {"bert": "abc"})


def result(phone_ids=None):
    phone_ids = [1, 2, 3] if phone_ids is None else phone_ids
    phones = ["k", "o", "."][: len(phone_ids)]
    return FrontendResult("こんにちは。", phones, phone_ids, None, torch.zeros(1024, len(phone_ids)))


def test_text_stage_saves_metadata_and_aligned_bert(tmp_path):
    frontend = FakeFrontend(result())
    stage_result = TextStage(frontend).run(record(), context(tmp_path))

    metadata = json.loads(stage_result.output_paths[0].read_text(encoding="utf-8"))
    bert = torch.load(stage_result.output_paths[1], weights_only=True)
    assert metadata["phone_ids"] == [1, 2, 3]
    assert metadata["bert_shape"] == [1024, 3]
    assert bert.shape == (1024, 3)
    assert stage_result.metadata["bert_dtype"] == "torch.float32"


def test_text_stage_calls_unified_frontend_once_for_mixed_text(tmp_path):
    frontend = FakeFrontend(result())
    TextStage(frontend).run(record("mixed", "你好。Hello."), context(tmp_path))
    assert frontend.calls == [("你好。Hello.", "mixed")]


def test_text_stage_turns_unsupported_language_and_empty_phones_into_sample_failure(tmp_path):
    stage = TextStage(FakeFrontend(result()))
    with pytest.raises(SampleFailure, match="unsupported language"):
        stage.run(record("ko"), context(tmp_path))

    empty = FakeFrontend(FrontendResult("", [], [], None, torch.zeros(1024, 0)))
    with pytest.raises(SampleFailure, match="no phones"):
        TextStage(empty).run(record(), context(tmp_path))


def test_text_stage_signature_tracks_language_text_and_frontend_assets(tmp_path):
    stage = TextStage(FakeFrontend(result()))
    first = stage.signature(record(), context(tmp_path))
    assert first != stage.signature(record(text="こんばんは。"), context(tmp_path))
    changed_assets = StageContext(
        context(tmp_path).experiment, V2PROPLUS, {}, {"bert": "different"}
    )
    assert first != stage.signature(record(), changed_assets)
