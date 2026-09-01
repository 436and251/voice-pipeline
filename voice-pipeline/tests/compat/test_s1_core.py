from __future__ import annotations

import os
from pathlib import Path

import pytest


def _tiny_config() -> dict[str, object]:
    return {
        "model": {
            "vocab_size": 17,
            "phoneme_vocab_size": 13,
            "embedding_dim": 16,
            "hidden_dim": 16,
            "head": 4,
            "linear_units": 64,
            "n_layer": 2,
            "dropout": 0,
            "EOS": 16,
            "random_bert": 0,
        }
    }


def test_s1_core_constructs_real_decoder_with_upstream_shapes() -> None:
    from voice_pipeline.core.gpt_sovits.s1 import build_s1_model

    model = build_s1_model(_tiny_config())

    assert tuple(model.ar_text_embedding.word_embeddings.weight.shape) == (13, 16)
    assert tuple(model.ar_audio_embedding.word_embeddings.weight.shape) == (17, 16)
    assert tuple(model.bert_proj.weight.shape) == (16, 1024)
    assert len(model.h.layers) == 2


def test_official_s1v3_checkpoint_loads_strictly() -> None:
    checkpoint_value = os.environ.get("VOICE_PIPELINE_TEST_S1_CHECKPOINT")
    if not checkpoint_value:
        pytest.skip("set VOICE_PIPELINE_TEST_S1_CHECKPOINT for the real-weight compatibility test")

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        pytest.fail(f"VOICE_PIPELINE_TEST_S1_CHECKPOINT is not a file: {checkpoint_path}")

    from voice_pipeline.core.gpt_sovits.compatibility.s1_checkpoint import load_s1_checkpoint

    model = load_s1_checkpoint(checkpoint_path, device="cpu")

    assert model.vocab_size == 1025
    assert model.phoneme_vocab_size == 732
    assert model.num_layers == 24
    assert tuple(model.ar_text_embedding.word_embeddings.weight.shape) == (732, 512)
    assert tuple(model.ar_audio_embedding.word_embeddings.weight.shape) == (1025, 512)
