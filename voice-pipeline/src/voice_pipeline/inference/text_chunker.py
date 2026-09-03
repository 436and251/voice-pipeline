from __future__ import annotations

import re


_DEFAULT_LIMITS = {"zh": 100, "ja": 500, "en": 500, "mixed": 500}
_PUNCTUATION_LEVELS = (frozenset("。！？.!?…"), frozenset("，、；：,;:"))
_PARAGRAPH_BREAK = re.compile(r"(?:\r?\n[ \t]*){2,}")


class TextChunker:
    def __init__(self, language: str, max_chars: int | None = None) -> None:
        if language not in _DEFAULT_LIMITS:
            raise ValueError("language must be zh, ja, en, or mixed")
        if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0):
            raise ValueError("max_chars must be a positive integer")
        self.max_chars = max_chars or _DEFAULT_LIMITS[language]

    def chunk(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is empty")
        chunks: list[str] = []
        for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.extend(self._split(paragraph, 0))
        return chunks

    def _split(self, text: str, level: int) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        if level == len(_PUNCTUATION_LEVELS):
            return [text[index : index + self.max_chars] for index in range(0, len(text), self.max_chars)]
        pieces = _split_at_punctuation(text, _PUNCTUATION_LEVELS[level])
        if len(pieces) == 1:
            return self._split(text, level + 1)
        resolved: list[str] = []
        siblings: list[str] = []
        for piece in pieces:
            if len(piece) > self.max_chars:
                resolved.extend(_pack(siblings, self.max_chars))
                siblings.clear()
                resolved.extend(self._split(piece, level + 1))
            else:
                siblings.append(piece)
        resolved.extend(_pack(siblings, self.max_chars))
        return resolved


def _split_at_punctuation(text: str, punctuation: frozenset[str]) -> list[str]:
    pieces: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in punctuation and not _is_decimal_point(text, index):
            end = index + 1
            while end < len(text) and text[end] in punctuation and not _is_decimal_point(text, end):
                end += 1
            pieces.append(text[start:end])
            start = end
            index = end
        else:
            index += 1
    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece]


def _is_decimal_point(text: str, index: int) -> bool:
    return (
        text[index] == "."
        and 0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _pack(pieces: list[str], limit: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > limit:
            packed.append(current)
            current = piece
        else:
            current += piece
    if current:
        packed.append(current)
    return packed


__all__ = ["TextChunker"]
