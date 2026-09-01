"""v2ProPlus ZH/JA/EN symbol table.

The ordering matches the ZH/JA/EN prefix of GPT-SoVITS `text/symbols2.py` at
upstream revision 48b1a0169a28582a8984402f82cf438d3bfa6aca. Korean and
Cantonese-only symbols are deliberately omitted because v2ProPlus scope here
is Chinese/Japanese/English; upstream appends those symbols after this prefix,
so shared phone IDs are unchanged.
"""

PUNCTUATION = ["!", "?", "…", ",", ".", "-"]
_SPECIAL = PUNCTUATION + ["SP", "SP2", "SP3", "UNK"]
_CONSONANTS = [
    "AA", "EE", "OO", "b", "c", "ch", "d", "f", "g", "h", "j", "k",
    "l", "m", "n", "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh",
]
_BASE_VOWELS = [
    "E", "En", "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er",
    "i", "i0", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "ir",
    "iu", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ui", "un",
    "uo", "v", "van", "ve", "vn",
]
_VOWELS = [f"{v}{tone}" for tone in range(1, 6) for v in _BASE_VOWELS]
_JAPANESE = [
    "I", "N", "U", "a", "b", "by", "ch", "cl", "d", "dy", "e", "f", "g",
    "gy", "h", "hy", "i", "j", "k", "ky", "m", "my", "n", "ny", "o", "p",
    "py", "r", "ry", "s", "sh", "t", "ts", "u", "v", "w", "y", "z",
]
_ARPA = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G",
    "AY0", "M", "AA0", "F", "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0",
    "EY1", "IH0", "K", "N", "W", "IY2", "T", "AA1", "ER1", "EH2", "OY0",
    "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER", "AW0", "UW0",
    "R", "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P", "AY1",
    "EY0", "OY2", "TH", "HH", "D", "ER0", "CH", "AO1", "AE1", "AO2", "OY1",
    "AY2", "IH1", "OW0", "L", "SH",
}

# Upstream sorts the shared set, then appends Japanese pitch-direction tokens.
SYMBOLS = tuple(sorted(set(["_"] + _CONSONANTS + _VOWELS + _JAPANESE + _SPECIAL + list(_ARPA))) + ["[", "]"])
_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}


def phone_ids(phones: list[str] | tuple[str, ...]) -> list[int]:
    """Convert already-cleaned GPT-SoVITS phone symbols to v2 phone IDs."""
    return [_SYMBOL_TO_ID[phone] for phone in phones]
