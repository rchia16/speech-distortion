from __future__ import annotations

import re
from typing import Dict, List, Protocol


class GraphemeToPhoneme(Protocol):
    """Converts transcript text into phones suitable for alignment and planning."""

    def phonemize(self, transcript: str) -> list[str]:
        raise NotImplementedError


class HeuristicEnglishG2P:
    """Small dependency-free English G2P for early prototype slices."""

    _LEXICON: Dict[str, List[str]] = {
        "yes": ["Y", "EH", "S"],
        "rabbit": ["R", "AE", "B", "IH", "T"],
        "cat": ["K", "AE", "T"],
        "blue": ["B", "L", "UW"],
        "spaghetti": ["S", "P", "AH", "G", "EH", "T", "IY"],
        "string": ["S", "T", "R", "IH", "NG"],
        "the": ["DH", "AH"],
        "come": ["K", "AH", "M"],
        "time": ["T", "AY", "M"],
    }

    _MULTI_CHAR_PHONES: List[tuple[str, List[str]]] = [
        ("tch", ["CH"]),
        ("dge", ["JH"]),
        ("igh", ["AY"]),
        ("sh", ["SH"]),
        ("ch", ["CH"]),
        ("th", ["TH"]),
        ("ph", ["F"]),
        ("ng", ["NG"]),
        ("qu", ["K", "W"]),
        ("ck", ["K"]),
        ("wh", ["W"]),
        ("ee", ["IY"]),
        ("oo", ["UW"]),
        ("ou", ["AW"]),
        ("ow", ["AW"]),
        ("oi", ["OY"]),
        ("oy", ["OY"]),
        ("ay", ["EY"]),
        ("ai", ["EY"]),
        ("ea", ["IY"]),
    ]

    _SINGLE_CHAR_PHONES: Dict[str, List[str]] = {
        "a": ["AE"],
        "b": ["B"],
        "c": ["K"],
        "d": ["D"],
        "e": ["EH"],
        "f": ["F"],
        "g": ["G"],
        "h": ["HH"],
        "i": ["IH"],
        "j": ["JH"],
        "k": ["K"],
        "l": ["L"],
        "m": ["M"],
        "n": ["N"],
        "o": ["OW"],
        "p": ["P"],
        "q": ["K"],
        "r": ["R"],
        "s": ["S"],
        "t": ["T"],
        "u": ["AH"],
        "v": ["V"],
        "w": ["W"],
        "x": ["K", "S"],
        "y": ["Y"],
        "z": ["Z"],
    }

    def phonemize(self, transcript: str) -> list[str]:
        phones: List[str] = []
        for word in re.findall(r"[A-Za-z']+", transcript):
            phones.extend(self.phonemize_word(word))
        return phones

    def phonemize_word(self, word: str) -> list[str]:
        lowered = re.sub(r"[^a-z']", "", word.lower())
        if not lowered:
            return []
        if lowered in self._LEXICON:
            return list(self._LEXICON[lowered])

        phones: List[str] = []
        index = 0
        while index < len(lowered):
            matched = False
            for chunk, chunk_phones in self._MULTI_CHAR_PHONES:
                if lowered.startswith(chunk, index):
                    phones.extend(chunk_phones)
                    index += len(chunk)
                    matched = True
                    break
            if matched:
                continue

            char = lowered[index]
            if char == "e" and index == len(lowered) - 1 and phones:
                index += 1
                continue
            phones.extend(self._SINGLE_CHAR_PHONES.get(char, []))
            index += 1

        if not phones:
            phones = ["AH"]
        return phones
