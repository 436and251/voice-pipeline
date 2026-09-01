"""Isolated upstream-derived Mandarin tone-sandhi primitives.

The training architecture does not depend on this module directly.  It is kept
separate from chinese.py so language rules cannot leak into orchestration code.
Heavy jieba/pypinyin-dependent rules are intentionally wired in a later frontend
part; the pure rules below match the pinned GPT-SoVITS implementation.
"""


class ToneSandhi:
    punc = "：，；。？！“”‘’':,;.?!"

    @staticmethod
    def _all_tone_three(finals: list[str]) -> bool:
        return all(item[-1] == "3" for item in finals)

    def _bu_sandhi(self, word: str, finals: list[str]) -> list[str]:
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            for index, char in enumerate(word):
                if char == "不" and index + 1 < len(word) and finals[index + 1][-1] == "4":
                    finals[index] = finals[index][:-1] + "2"
        return finals

    def _yi_sandhi(self, word: str, finals: list[str]) -> list[str]:
        if "一" in word and all(item.isnumeric() for item in word if item != "一"):
            return finals
        if len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            finals[1] = finals[1][:-1] + "5"
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for index, char in enumerate(word):
                if char != "一" or index + 1 >= len(word):
                    continue
                if finals[index + 1][-1] == "4":
                    finals[index] = finals[index][:-1] + "2"
                elif word[index + 1] not in self.punc:
                    finals[index] = finals[index][:-1] + "4"
        return finals

    def _three_sandhi(self, word: str, finals: list[str]) -> list[str]:
        if len(word) == 2 and self._all_tone_three(finals):
            finals[0] = finals[0][:-1] + "2"
        elif len(word) == 4:
            merged: list[str] = []
            for sub in (finals[:2], finals[2:]):
                if self._all_tone_three(sub):
                    sub[0] = sub[0][:-1] + "2"
                merged.extend(sub)
            finals = merged
        return finals

    @staticmethod
    def _merge_bu(seg):
        new_seg = []
        last_word = ""
        for word, pos in seg:
            if last_word == "不":
                word = last_word + word
            if word != "不":
                new_seg.append((word, pos))
            last_word = word[:]
        if last_word == "不":
            new_seg.append((last_word, "d"))
        return new_seg

    @staticmethod
    def _merge_yi(seg):
        new_seg = []
        i = 0
        while i < len(seg):
            word, pos = seg[i]
            merged = False
            if i - 1 >= 0 and word == "一" and i + 1 < len(seg):
                last = new_seg[-1] if new_seg else seg[i - 1]
                if last[0] == seg[i + 1][0] and last[1] == "v" and seg[i + 1][1] == "v":
                    new_seg[-1] = [last[0] + "一" + seg[i + 1][0], last[1]]
                    i += 2
                    merged = True
            if not merged:
                new_seg.append([word, pos])
                i += 1
        seg = new_seg
        new_seg = []
        for word, pos in seg:
            if new_seg and new_seg[-1][0] == "一":
                new_seg[-1][0] += word
            else:
                new_seg.append([word, pos])
        return new_seg

    @staticmethod
    def _merge_er(seg):
        new_seg = []
        for index, (word, pos) in enumerate(seg):
            if index - 1 >= 0 and word == "儿" and seg[index - 1][0] != "#":
                new_seg[-1][0] += word
            else:
                new_seg.append([word, pos])
        return new_seg

    @staticmethod
    def _merge_reduplication(seg):
        new_seg = []
        for word, pos in seg:
            if new_seg and word == new_seg[-1][0]:
                new_seg[-1][0] += word
            else:
                new_seg.append([word, pos])
        return new_seg
