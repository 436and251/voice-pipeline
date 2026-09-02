import os
from pathlib import Path

import pytest
import torch


def _base_s2g() -> Path:
    root = os.environ.get("VOICE_PIPELINE_TEST_S2_DIR")
    if not root:
        pytest.skip("set VOICE_PIPELINE_TEST_S2_DIR for real semantic extraction")
    path = Path(root) / "s2Gv2ProPlus.pth"
    if not path.is_file():
        pytest.fail(f"missing real base S2G: {path}")
    return path


def test_semantic_extractor_matches_direct_base_s2g_extraction():
    from voice_pipeline.training.preprocess.semantic_stage import SemanticExtractor

    extractor = SemanticExtractor(_base_s2g(), "cpu", "fp32")
    ssl = torch.linspace(-1, 1, 768 * 8).reshape(1, 768, 8)
    tokens = extractor.extract(ssl)
    with torch.inference_mode():
        direct = extractor.model.extract_latent(ssl)[0, 0].to(dtype=torch.long)

    assert torch.equal(tokens, direct)
    assert tokens.ndim == 1 and tokens.numel() > 0
    assert tokens.min() >= 0 and tokens.max() <= 1023
