from voice_pipeline.core.gpt_sovits.frontend.japanese_runtime import JapaneseFrontend


def test_japanese_runtime_uses_pyopenjtalk_prosody() -> None:
    normalized, phones, word2ph = JapaneseFrontend().process("こんにちは。")

    assert normalized == "こんにちは。"
    assert phones == ["k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "."]
    assert word2ph is None


def test_japanese_runtime_collapses_only_upstream_punctuation() -> None:
    normalized, _, _ = JapaneseFrontend().process("こんにちは!!!今日は???晴れ..")

    assert normalized == "こんにちは!今日は?晴れ."


def test_japanese_runtime_expands_full_width_percent_for_g2p() -> None:
    normalized, phones, _ = JapaneseFrontend().process("50％")

    assert normalized == "50％"
    assert "%" not in phones
    assert "％" not in phones
