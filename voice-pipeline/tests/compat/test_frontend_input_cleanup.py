from voice_pipeline.core.gpt_sovits.frontend.multilingual import sanitize_frontend_text


def test_shared_frontend_applies_official_training_symbol_cleanup():
    assert sanitize_frontend_text("成功率50%￥500") == "成功率50-,500"


def test_cleanup_leaves_other_text_unchanged():
    assert sanitize_frontend_text("こんにちは。Hello!") == "こんにちは。Hello!"
