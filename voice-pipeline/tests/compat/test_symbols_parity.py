import pytest

from voice_pipeline.core.gpt_sovits.frontend.symbols import SYMBOLS, phone_ids


def test_v2proplus_zh_ja_en_prefix_size_matches_upstream_revision():
    # Reconstructed directly from symbols2.py at the pinned upstream revision:
    # sorted(shared ZH/JA/EN set) + Japanese pitch tokens '[' and ']'.
    assert len(SYMBOLS) == 324


def test_core_phone_ids_match_upstream_v2_order():
    assert phone_ids(["!", "AA", "a1", "N", "AH0", "UNK", "[", "]"]) == [
        0, 5, 97, 64, 12, 86, 322, 323
    ]


def test_unknown_phone_is_not_silently_remapped():
    with pytest.raises(KeyError):
        phone_ids(["NOT_A_GPT_SOVITS_PHONE"])
