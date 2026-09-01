from voice_pipeline.core.gpt_sovits.frontend.chinese import (
    g2p_segments,
    split_g2p_sentences,
)


class FakeToneModifier:
    def __init__(self):
        self.pre_merge_inputs = []
        self.modified = []

    def pre_merge_for_modify(self, seg):
        self.pre_merge_inputs.append(list(seg))
        return list(seg)

    def modified_tone(self, word, pos, finals):
        self.modified.append((word, pos, list(finals)))
        return list(finals)


class FakeG2PW:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def batch(self, texts):
        self.calls.append(list(texts))
        return self.outputs


def fake_segmenter(text):
    table = {
        "你好,": [("你", "r"), ("好", "a"), (",", "x")],
        "世界!": [("世界", "n"), ("!", "x")],
        "中A文。": [("中", "n"), ("A", "eng"), ("文", "n"), ("。", "x")],
    }
    return table[text]


def fake_converter(token):
    table = {
        "ni3": ("n", "i3"),
        "hao3": ("h", "ao3"),
        "shi4": ("sh", "i4"),
        "jie4": ("j", "ie4"),
        "zhong1": ("zh", "ong1"),
        "wen2": ("w", "en2"),
        ",": (",", ","),
        "!": ("!", "!"),
        ".": (".", "."),
    }
    return table[token]


def identity_corrector(word, pinyins):
    return pinyins


def test_split_g2p_sentences_preserves_punctuation_boundaries():
    assert split_g2p_sentences("你好,世界!再见") == ["你好,", "世界!", "再见"]


def test_g2pw_batches_nonempty_ascii_stripped_segments_and_keeps_cursor_alignment():
    tone = FakeToneModifier()
    g2pw = FakeG2PW([
        ["ni3", "hao3", ","],
        ["shi4", "jie4", "!"],
    ])

    phones, word2ph = g2p_segments(
        ["你好,", "ABC", "世界!"],
        segmenter=fake_segmenter,
        tone_modifier=tone,
        pinyin_provider=g2pw,
        pinyin_converter=fake_converter,
        pronunciation_corrector=identity_corrector,
    )

    assert g2pw.calls == [["你好,", "世界!"]]
    assert phones == ["n", "i3", "h", "ao3", ",", "sh", "ir4", "j", "ie4", "!"]
    assert word2ph == [2, 2, 1, 2, 2, 1]
    assert sum(word2ph) == len(phones)


def test_eng_pos_advances_character_cursor_but_does_not_emit_phones():
    tone = FakeToneModifier()
    g2pw = FakeG2PW([["zhong1", "ignored", "wen2", "."]])

    phones, word2ph = g2p_segments(
        ["中A文。"],
        segmenter=fake_segmenter,
        tone_modifier=tone,
        pinyin_provider=g2pw,
        pinyin_converter=fake_converter,
        pronunciation_corrector=identity_corrector,
        strip_ascii=False,
    )

    assert phones == ["zh", "ong1", "w", "en2", "."]
    assert word2ph == [2, 2, 1]
    assert [item[0] for item in tone.modified] == ["中", "文", "。"]


def test_pronunciation_correction_happens_before_tone_modification():
    events = []

    class Tone(FakeToneModifier):
        def modified_tone(self, word, pos, finals):
            events.append(("tone", word, list(finals)))
            return list(finals)

    def corrector(word, pinyins):
        events.append(("correct", word, list(pinyins)))
        return pinyins

    g2p_segments(
        ["你好,"],
        segmenter=fake_segmenter,
        tone_modifier=Tone(),
        pinyin_provider=FakeG2PW([["ni3", "hao3", ","]]),
        pinyin_converter=fake_converter,
        pronunciation_corrector=corrector,
    )

    assert events == [
        ("correct", "你", ["ni3"]),
        ("tone", "你", ["i3"]),
        ("correct", "好", ["hao3"]),
        ("tone", "好", ["ao3"]),
        ("correct", ",", [","]),
        ("tone", ",", [","]),
    ]


def test_public_g2p_splits_then_runs_same_orchestration():
    from voice_pipeline.core.gpt_sovits.frontend.chinese import g2p

    tone = FakeToneModifier()
    phones, word2ph = g2p(
        "你好,世界!",
        segmenter=fake_segmenter,
        tone_modifier=tone,
        pinyin_provider=FakeG2PW([
            ["ni3", "hao3", ","],
            ["shi4", "jie4", "!"],
        ]),
        pinyin_converter=fake_converter,
        pronunciation_corrector=identity_corrector,
    )

    assert phones == ["n", "i3", "h", "ao3", ",", "sh", "ir4", "j", "ie4", "!"]
    assert word2ph == [2, 2, 1, 2, 2, 1]
