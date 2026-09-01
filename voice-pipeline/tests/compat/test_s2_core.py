from __future__ import annotations

import os
from pathlib import Path

import pytest


def _weight_path(name: str) -> Path:
    root_value = os.environ.get("VOICE_PIPELINE_TEST_S2_DIR")
    if not root_value:
        pytest.skip("set VOICE_PIPELINE_TEST_S2_DIR for real-weight S2 compatibility tests")
    path = Path(root_value) / name
    if not path.is_file():
        pytest.fail(f"missing real-weight S2 test asset: {path}")
    return path


def test_v2proplus_discriminator_has_official_eight_branches() -> None:
    from voice_pipeline.core.gpt_sovits.s2_v2proplus import build_s2_discriminator

    model = build_s2_discriminator()

    assert len(model.discriminators) == 8
    assert [disc.period for disc in model.discriminators[1:]] == [2, 3, 5, 7, 11, 17, 23]


def test_official_s2g_v2proplus_checkpoint_loads_strictly() -> None:
    from voice_pipeline.core.gpt_sovits.compatibility.s2_checkpoint import load_s2_generator

    model = load_s2_generator(_weight_path("s2Gv2ProPlus.pth"), device="cpu")

    assert model.version == "v2ProPlus"
    assert model.is_v2pro is True
    assert model.freeze_quantizer is True
    assert tuple(model.enc_p.text_embedding.weight.shape) == (732, 192)
    assert tuple(model.sv_emb.weight.shape) == (1024, 20480)


def test_official_s2d_v2proplus_checkpoint_loads_strictly() -> None:
    from voice_pipeline.core.gpt_sovits.compatibility.s2_checkpoint import load_s2_discriminator

    model = load_s2_discriminator(_weight_path("s2Dv2ProPlus.pth"), device="cpu")

    assert len(model.discriminators) == 8
    assert [disc.period for disc in model.discriminators[1:]] == [2, 3, 5, 7, 11, 17, 23]
