from voice_pipeline.core.gpt_sovits.frontend.tone_sandhi import ToneSandhi


def test_bu_sandhi_before_tone4_becomes_tone2():
    t = ToneSandhi()
    assert t._bu_sandhi("不怕", ["u4", "a4"]) == ["u2", "a4"]


def test_yi_sandhi_matches_upstream_rules():
    t = ToneSandhi()
    assert t._yi_sandhi("一天", ["i1", "ian1"]) == ["i4", "ian1"]
    assert t._yi_sandhi("一段", ["i1", "uan4"]) == ["i2", "uan4"]
    assert t._yi_sandhi("看一看", ["an4", "i1", "an4"]) == ["an4", "i5", "an4"]


def test_two_third_tones_change_first_to_tone2():
    t = ToneSandhi()
    assert t._three_sandhi("你好", ["i3", "ao3"]) == ["i2", "ao3"]


def test_merge_helpers_match_upstream():
    t = ToneSandhi()
    assert t._merge_bu([("不", "d"), ("怕", "v")]) == [("不怕", "v")]
    assert t._merge_yi([("听", "v"), ("一", "m"), ("听", "v")]) == [["听一听", "v"]]
    assert t._merge_er([("小院", "n"), ("儿", "n")]) == [["小院儿", "n"]]
