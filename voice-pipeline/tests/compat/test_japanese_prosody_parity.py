import pytest

from voice_pipeline.core.gpt_sovits.frontend import japanese


def _label(phone: str, *, a1: int = 1, a2: int = 1, a3: int = 0, f1: int = 3, e3: int = 0) -> str:
    return f"xx-{phone}+xx/A:{a1}+{a2}+{a3}/F:{f1}_xx!{e3}_"


def test_numeric_feature_parser_matches_upstream_regex_behavior():
    label = _label("k", a1=-1, a2=2, a3=1, f1=4)
    assert japanese._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", label) == -1
    assert japanese._numeric_feature_by_regex(r"\+(\d+)\+", label) == 2
    assert japanese._numeric_feature_by_regex(r"\+(\d+)/", label) == 1
    assert japanese._numeric_feature_by_regex(r"/F:(\d+)_", label) == 4
    assert japanese._numeric_feature_by_regex(r"/MISSING:(\d+)", label) == -50


def test_labels_to_prosody_marks_start_pause_and_statement_end():
    labels = [
        _label("sil"),
        _label("k", a2=1),
        _label("pau"),
        _label("o", a2=2),
        _label("sil", e3=0),
    ]
    phones = japanese._labels_to_prosody(labels)
    assert phones[0] == "^"
    assert "_" in phones
    assert phones[-1] == "$"


def test_labels_to_prosody_marks_question_end():
    labels = [_label("sil"), _label("a"), _label("sil", e3=1)]
    assert japanese._labels_to_prosody(labels)[-1] == "?"


def test_labels_to_prosody_emits_rise_fall_and_phrase_boundary():
    rising = [
        _label("sil"),
        _label("o", a1=1, a2=1, a3=0, f1=3),
        _label("k", a1=1, a2=2, a3=0, f1=3),
        _label("sil"),
    ]
    falling = [
        _label("sil"),
        _label("o", a1=0, a2=1, a3=0, f1=3),
        _label("k", a1=1, a2=2, a3=0, f1=3),
        _label("sil"),
    ]
    boundary = [
        _label("sil"),
        _label("o", a1=1, a2=1, a3=1, f1=3),
        _label("k", a1=1, a2=1, a3=0, f1=3),
        _label("sil"),
    ]
    assert "[" in japanese._labels_to_prosody(rising)
    assert "]" in japanese._labels_to_prosody(falling)
    assert "#" in japanese._labels_to_prosody(boundary)


def test_real_pyopenjtalk_documented_example_when_dependency_is_available():
    pytest.importorskip("pyopenjtalk")
    assert japanese.pyopenjtalk_g2p_prosody("こんにちは。") == [
        "^", "k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "$"
    ]


def test_japanese_percent_symbol_replacement_matches_upstream():
    assert japanese.symbols_to_japanese("50％") == "50パーセント"


def test_preprocess_japanese_keeps_marks_and_ignores_spaces(monkeypatch):
    class FakeOpenJTalk:
        @staticmethod
        def g2p(text):
            return "k o N n i ch i w a"

    monkeypatch.setitem(__import__("sys").modules, "pyopenjtalk", FakeOpenJTalk)
    assert japanese.preprocess_jap("こんにちは ！", with_prosody=False) == [
        "k", "o", "N", "n", "i", "ch", "i", "w", "a", "！"
    ]


def test_g2p_applies_japanese_phone_punctuation_mapping(monkeypatch):
    monkeypatch.setattr(japanese, "preprocess_jap", lambda text, with_prosody=True: ["k", "o", "！"])
    assert japanese.g2p("ignored") == ["k", "o", "!"]


def test_real_japanese_g2p_when_pyopenjtalk_is_available():
    pytest.importorskip("pyopenjtalk")
    assert japanese.g2p("こんにちは。") == ["k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "."]
