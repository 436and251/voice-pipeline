"""Japanese normalization and Open JTalk G2P preserved from GPT-SoVITS."""

import re

from .symbols import PUNCTUATION

_REPLACEMENTS = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
}


def post_replace_ph(phone: str) -> str:
    return _REPLACEMENTS.get(phone, phone)


def text_normalize(text: str) -> str:
    punctuations = "".join(re.escape(p) for p in PUNCTUATION)
    return re.sub(f"([{punctuations}])([{punctuations}])+", r"\1", text)


def _numeric_feature_by_regex(regex: str, label: str) -> int:
    match = re.search(regex, label)
    return -50 if match is None else int(match.group(1))


def _labels_to_prosody(labels: list[str], drop_unvoiced_vowels: bool = True) -> list[str]:
    """Convert Open JTalk full-context labels to GPT-SoVITS prosody tokens."""
    phones: list[str] = []
    last = len(labels) - 1
    for index, label in enumerate(labels):
        phone = re.search(r"\-(.*?)\+", label).group(1)
        if drop_unvoiced_vowels and phone in "AEIOU":
            phone = phone.lower()

        if phone == "sil":
            if index == 0:
                phones.append("^")
            elif index == last:
                ending = _numeric_feature_by_regex(r"!(\d+)_", label)
                if ending == 0:
                    phones.append("$")
                elif ending == 1:
                    phones.append("?")
            continue
        if phone == "pau":
            phones.append("_")
            continue

        phones.append(phone)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", label)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", label)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", label)
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", label)
        next_a2 = _numeric_feature_by_regex(r"\+(\d+)\+", labels[index + 1])

        if a3 == 1 and next_a2 == 1 and phone in "aeiouAEIOUNcl":
            phones.append("#")
        elif a1 == 0 and next_a2 == a2 + 1 and a2 != f1:
            phones.append("]")
        elif a2 == 1 and next_a2 == 2:
            phones.append("[")
    return phones


def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> list[str]:
    """Match GPT-SoVITS' Open JTalk prosody extraction for Japanese text."""
    import pyopenjtalk

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    return _labels_to_prosody(labels, drop_unvoiced_vowels)

_JAPANESE_CHARACTERS = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)
_JAPANESE_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)
_SYMBOL_REPLACEMENTS = ((re.compile("％"), "パーセント"),)


def symbols_to_japanese(text: str) -> str:
    for regex, replacement in _SYMBOL_REPLACEMENTS:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text: str, with_prosody: bool = False) -> list[str]:
    import pyopenjtalk

    text = symbols_to_japanese(text).lower()
    sentences = re.split(_JAPANESE_MARKS, text)
    marks = re.findall(_JAPANESE_MARKS, text)
    phones: list[str] = []
    for index, sentence in enumerate(sentences):
        if re.match(_JAPANESE_CHARACTERS, sentence):
            if with_prosody:
                phones.extend(pyopenjtalk_g2p_prosody(sentence)[1:-1])
            else:
                phones.extend(pyopenjtalk.g2p(sentence).split(" "))
        if index < len(marks):
            if marks[index] == " ":
                continue
            phones.append(marks[index].replace(" ", ""))
    return phones


def g2p(norm_text: str, with_prosody: bool = True) -> list[str]:
    return [post_replace_ph(phone) for phone in preprocess_jap(norm_text, with_prosody)]
