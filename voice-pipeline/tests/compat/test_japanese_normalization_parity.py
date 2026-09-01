from voice_pipeline.core.gpt_sovits.frontend.japanese import post_replace_ph, text_normalize


def test_japanese_ascii_repeated_punctuation_matches_upstream():
    assert text_normalize("こんにちは!!!今日は???晴れ..") == "こんにちは!今日は?晴れ."


def test_japanese_full_width_punctuation_is_left_for_phone_postprocessing():
    assert text_normalize("こんにちは！！！") == "こんにちは！！！"


def test_japanese_phone_punctuation_replacement_matches_upstream():
    assert [post_replace_ph(x) for x in ["：", "；", "，", "。", "！", "？", "、", "..."]] == [
        ",", ",", ",", ".", "!", "?", ",", "…"
    ]
