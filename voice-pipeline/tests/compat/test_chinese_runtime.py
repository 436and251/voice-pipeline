from __future__ import annotations

import os
from pathlib import Path

import pytest


def _asset(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"set {env_name} for real Chinese frontend tests")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"missing real Chinese frontend asset: {path}")
    return path


def test_real_g2pw_disambiguates_chongqing() -> None:
    from voice_pipeline.core.gpt_sovits.frontend.chinese_runtime import ChineseFrontend

    frontend = ChineseFrontend(
        _asset("VOICE_PIPELINE_TEST_G2PW_DIR"),
        _asset("VOICE_PIPELINE_TEST_BERT_DIR"),
    )

    normalized, phones, word2ph = frontend.process("重庆。")

    assert normalized == "重庆."
    assert phones[:4] == ["ch", "ong2", "q", "ing4"]
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(normalized)


def test_missing_g2pw_directory_fails_without_downloading(tmp_path: Path) -> None:
    from voice_pipeline.core.gpt_sovits.frontend.chinese_runtime import ChineseFrontend

    missing = tmp_path / "G2PWModel"

    with pytest.raises(FileNotFoundError, match="G2PWModel"):
        ChineseFrontend(missing, _asset("VOICE_PIPELINE_TEST_BERT_DIR"))

    assert not missing.exists()
