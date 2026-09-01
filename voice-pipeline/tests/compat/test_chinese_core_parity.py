from voice_pipeline.core.gpt_sovits.frontend.chinese import (
    _merge_erhua,
    phones_from_initials_finals,
    replace_consecutive_punctuation,
    replace_punctuation,
)


def test_replace_punctuation_matches_upstream_cleanup():
    assert replace_punctuation("嗯呣，A1～好/吗？") == "恩母,…好,吗?"


def test_consecutive_mixed_punctuation_collapses_to_first_mark():
    assert replace_consecutive_punctuation("你好!!!??世界") == "你好!世界"


def test_must_erhua_inherits_previous_syllable_tone():
    initials, finals = _merge_erhua(
        ["x", "f", ""], ["i3", "u5", "er2"], "媳妇儿", "n"
    )
    assert initials == ["x", "f", ""]
    assert finals == ["i3", "u5", "er5"]


def test_not_erhua_word_remains_separate_er_syllable():
    initials, finals = _merge_erhua(["n", ""], ["v3", "er2"], "女儿", "n")
    assert initials == ["n", ""]
    assert finals == ["v3", "er2"]


def test_phone_mapping_preserves_upstream_pinyin_rewrites_and_word2ph():
    phones, word2ph = phones_from_initials_finals(
        initials=["sh", "", ","],
        finals=["uei3", "ing2", ","],
        segment="水英，",
    )
    assert phones == ["sh", "ui3", "y", "ing2", ","]
    assert word2ph == [2, 2, 1]
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len("水英，")


def test_er1_is_normalized_to_er2_before_not_erhua_guard():
    _, finals = _merge_erhua(["n", ""], ["v3", "er1"], "女儿", "n")
    assert finals == ["v3", "er2"]


def test_full_opencpop_table_maps_into_v2proplus_symbol_vocabulary():
    from voice_pipeline.core.gpt_sovits.frontend.chinese import _pinyin_map
    from voice_pipeline.core.gpt_sovits.frontend.symbols import SYMBOLS

    mapping = _pinyin_map()
    assert len(mapping) == 429
    for consonant, vowel in (value.split(" ") for value in mapping.values()):
        assert consonant in SYMBOLS
        for tone in "12345":
            assert vowel + tone in SYMBOLS
