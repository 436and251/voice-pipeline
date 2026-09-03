import pytest

from voice_pipeline.inference.text_chunker import TextChunker


@pytest.mark.parametrize(
    ("language", "limit"),
    [("zh", 100), ("ja", 500), ("en", 500), ("mixed", 500)],
)
def test_language_default_is_the_hard_limit(language: str, limit: int):
    text = "字" * (limit + 1)
    chunks = TextChunker(language).chunk(text)
    assert [len(chunk) for chunk in chunks] == [limit, 1]
    assert "".join(chunks) == text


def test_recursively_prefers_strong_then_weak_punctuation():
    strong_text = "甲" * 60 + "。" + "乙" * 60 + "！"
    assert TextChunker("zh").chunk(strong_text) == ["甲" * 60 + "。", "乙" * 60 + "！"]

    weak_text = "甲" * 40 + "，" + "乙" * 40 + "；" + "丙" * 40 + "。"
    assert TextChunker("zh").chunk(weak_text) == ["甲" * 40 + "，" + "乙" * 40 + "；", "丙" * 40 + "。"]


def test_recursive_child_is_not_repacked_with_next_strong_sentence():
    text = "甲" * 120 + "。" + "乙" * 10 + "。"
    assert TextChunker("zh").chunk(text) == ["甲" * 100, "甲" * 20 + "。", "乙" * 10 + "。"]


def test_paragraphs_are_never_merged():
    assert TextChunker("zh").chunk("第一段。\n\n第二段。") == ["第一段。", "第二段。"]


def test_english_decimal_point_is_not_a_sentence_boundary():
    text = "a" * 250 + "3.14" + "b" * 250 + "."
    chunks = TextChunker("en").chunk(text)
    assert [len(chunk) for chunk in chunks] == [500, 5]
    assert "".join(chunks) == text


def test_custom_limit_and_inputs_are_validated():
    assert TextChunker("ja", max_chars=3).chunk("abcdef") == ["abc", "def"]
    for language in ("auto", "fr", ""):
        with pytest.raises(ValueError, match="language"):
            TextChunker(language)
    for limit in (0, -1, True):
        with pytest.raises(ValueError, match="max_chars"):
            TextChunker("zh", max_chars=limit)
    with pytest.raises(ValueError, match="empty"):
        TextChunker("zh").chunk(" \n\t")
