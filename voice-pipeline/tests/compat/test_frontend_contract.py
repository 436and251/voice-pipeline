from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def _bert_path() -> Path:
    value = os.environ.get("VOICE_PIPELINE_TEST_BERT_DIR")
    if not value:
        pytest.skip("set VOICE_PIPELINE_TEST_BERT_DIR for real BERT compatibility tests")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"missing real BERT test asset: {path}")
    return path


def test_frontend_result_rejects_misaligned_bert_columns() -> None:
    from voice_pipeline.core.gpt_sovits.frontend.contract import FrontendResult

    with pytest.raises(ValueError, match="BERT columns"):
        FrontendResult(
            normalized_text="a",
            phones=["AA"],
            phone_ids=[5],
            word2ph=None,
            bert_features=torch.zeros(1024, 2),
        )


def test_character_features_repeat_by_literal_word2ph() -> None:
    from voice_pipeline.core.gpt_sovits.frontend.bert import expand_character_features

    characters = torch.tensor([[1.0, 10.0], [2.0, 20.0]])

    expanded = expand_character_features(characters, [2, 1])

    assert torch.equal(expanded, torch.tensor([[1.0, 1.0, 2.0], [10.0, 10.0, 20.0]]))


def test_real_chinese_bert_aligns_characters_to_phones() -> None:
    from voice_pipeline.core.gpt_sovits.frontend.bert import BertAligner

    aligner = BertAligner(_bert_path(), device="cpu")

    features = aligner.extract("你好", [2, 2])

    assert features.shape == (1024, 4)
    assert features.dtype == torch.float32
    assert features.device.type == "cpu"
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features) > 0
