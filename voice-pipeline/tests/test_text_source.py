from pathlib import Path

import pytest

from voice_pipeline.inference.text_source import resolve_text_source


def test_resolves_exactly_one_inline_or_file_source(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("file text", encoding="utf-8")

    assert resolve_text_source("  inline text\n", None) == "inline text"
    assert resolve_text_source(None, path) == "file text"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_text_source("inline", path)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_text_source(None, None)


def test_reads_utf8_sig_txt_without_leaking_bom(tmp_path: Path):
    path = tmp_path / "日文.TXT"
    path.write_text("\ufeff今日はいい天気です。\n", encoding="utf-8")
    assert resolve_text_source(None, path) == "今日はいい天気です。"


@pytest.mark.parametrize("content", [b"", b"  \r\n\t"])
def test_rejects_empty_txt(tmp_path: Path, content: bytes):
    path = tmp_path / "empty.txt"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="empty"):
        resolve_text_source(None, path)


def test_rejects_non_txt_missing_and_invalid_utf8(tmp_path: Path):
    markdown = tmp_path / "input.md"
    markdown.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match=".txt"):
        resolve_text_source(None, markdown)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_text_source(None, tmp_path / "missing.txt")

    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        resolve_text_source(None, invalid)
