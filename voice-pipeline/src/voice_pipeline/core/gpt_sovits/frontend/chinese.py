"""Pure Chinese frontend primitives preserved from GPT-SoVITS chinese2.py."""

from functools import lru_cache
from pathlib import Path
import re

from .symbols import PUNCTUATION

_REP_MAP = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": ".", "·": ",", "、": ",", "...": "…", "$": ".", "/": ",",
    "—": "-", "~": "…", "～": "…",
}
_MUST_ERHUA = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
_NOT_ERHUA = {
    "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
    "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
    "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿",
    "可儿", "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿",
    "鸟儿", "猪儿", "猫儿", "狗儿", "少儿",
}


def replace_punctuation(text: str) -> str:
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(mark) for mark in _REP_MAP))
    text = pattern.sub(lambda match: _REP_MAP[match.group()], text)
    allowed = "".join(PUNCTUATION)
    return re.sub(r"[^\u4e00-\u9fa5" + allowed + r"]+", "", text)


def replace_consecutive_punctuation(text: str) -> str:
    marks = "".join(re.escape(mark) for mark in PUNCTUATION)
    return re.sub(f"([{marks}])([{marks}])+", r"\1", text)


def _merge_erhua(
    initials: list[str], finals: list[str], word: str, pos: str
) -> tuple[list[str], list[str]]:
    for index, final in enumerate(finals):
        if index == len(finals) - 1 and word[index] == "儿" and final == "er1":
            finals[index] = "er2"

    if word not in _MUST_ERHUA and (word in _NOT_ERHUA or pos in {"a", "j", "nr"}):
        return initials, finals
    if len(finals) != len(word):
        return initials, finals

    new_initials: list[str] = []
    new_finals: list[str] = []
    for index, final in enumerate(finals):
        if (
            index == len(finals) - 1
            and word[index] == "儿"
            and final in {"er2", "er5"}
            and word[-2:] not in _NOT_ERHUA
            and new_finals
        ):
            final = "er" + new_finals[-1][-1]
        new_initials.append(initials[index])
        new_finals.append(final)
    return new_initials, new_finals


@lru_cache(maxsize=1)
def _pinyin_map() -> dict[str, str]:
    path = Path(__file__).with_name("opencpop-strict.txt")
    return dict(line.rstrip().split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines())


def phones_from_initials_finals(
    initials: list[str], finals: list[str], segment: str
) -> tuple[list[str], list[int]]:
    """Map tone-marked pinyin components to GPT-SoVITS phones and word2ph."""
    phones: list[str] = []
    word2ph: list[int] = []
    mapping = _pinyin_map()

    for initial, final in zip(initials, finals):
        raw_pinyin = initial + final
        if initial == final:
            if initial not in PUNCTUATION:
                raise AssertionError(initial)
            phone = [initial]
        else:
            base, tone = final[:-1], final[-1]
            if tone not in "12345":
                raise AssertionError(raw_pinyin)
            pinyin = initial + base
            if initial:
                base = {"uei": "ui", "iou": "iu", "uen": "un"}.get(base, base)
                pinyin = initial + base
            else:
                pinyin = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}.get(pinyin, pinyin)
                if pinyin not in mapping and pinyin:
                    pinyin = {"v": "yu", "e": "e", "i": "y", "u": "w"}.get(pinyin[0], pinyin[0]) + pinyin[1:]
            if pinyin not in mapping:
                raise AssertionError((pinyin, segment, raw_pinyin))
            consonant, vowel = mapping[pinyin].split(" ")
            phone = [consonant, vowel + tone]
        phones.extend(phone)
        word2ph.append(len(phone))

    return phones, word2ph


def split_g2p_sentences(text: str) -> list[str]:
    """Split exactly as upstream chinese2.g2p: punctuation stays on the left."""
    pattern = rf"(?<=[{re.escape(''.join(PUNCTUATION))}])\s*"
    return [part for part in re.split(pattern, text) if part.strip()]


def g2p_segments(
    segments: list[str],
    *,
    segmenter,
    tone_modifier,
    pinyin_provider,
    pinyin_converter,
    pronunciation_corrector,
    strip_ascii: bool = True,
) -> tuple[list[str], list[int]]:
    """Run the upstream chinese2 G2PW orchestration with injectable dependencies.

    Injection keeps this algorithm independently testable; production adapters for
    jieba/G2PW/pypinyin are wired separately.
    """
    processed = [re.sub(r"[a-zA-Z]+", "", segment) if strip_ascii else segment for segment in segments]
    nonempty = [segment for segment in processed if segment]
    batch_results = pinyin_provider.batch(nonempty) if nonempty else []
    batch_cursor = 0
    all_phones: list[str] = []
    all_word2ph: list[int] = []

    for segment in processed:
        if not segment:
            continue
        pinyins = batch_results[batch_cursor]
        batch_cursor += 1
        words = tone_modifier.pre_merge_for_modify(segmenter(segment))
        initials: list[str] = []
        finals: list[str] = []
        char_cursor = 0

        for word, pos in words:
            next_cursor = char_cursor + len(word)
            if pos == "eng":
                char_cursor = next_cursor
                continue
            word_pinyins = pronunciation_corrector(word, pinyins[char_cursor:next_cursor])
            word_initials: list[str] = []
            word_finals: list[str] = []
            for token in word_pinyins:
                initial, final = pinyin_converter(token)
                word_initials.append(initial)
                word_finals.append(final)
            char_cursor = next_cursor
            word_finals = tone_modifier.modified_tone(word, pos, word_finals)
            word_initials, word_finals = _merge_erhua(word_initials, word_finals, word, pos)
            initials.extend(word_initials)
            finals.extend(word_finals)

        phones, word2ph = phones_from_initials_finals(initials, finals, segment)
        all_phones.extend(phones)
        all_word2ph.extend(word2ph)

    return all_phones, all_word2ph


def g2p(text: str, **dependencies) -> tuple[list[str], list[int]]:
    """Public Chinese G2P entry preserving upstream sentence splitting semantics."""
    return g2p_segments(split_g2p_sentences(text), **dependencies)
