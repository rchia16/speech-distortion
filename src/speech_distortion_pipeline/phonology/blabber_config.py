from __future__ import annotations

import os
from typing import Final


DEFAULT_GUI_INPUT_FILENAME: Final[str] = "yes_slow.wav"

COQUI_ENV_NAME: Final[str] = os.environ.get("BLABBER_CONDA_ENV", "coqui-blabber")
COQUI_MODEL_NAME: Final[str] = os.environ.get(
    "BLABBER_COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph"
)
COQUI_CLONE_MODEL_NAME: Final[str] = os.environ.get(
    "BLABBER_COQUI_CLONE_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
)

PHONE_SUBSTITUTIONS: Final[dict[str, tuple[str, ...]]] = {
    "R": ("W", "L"),
    "L": ("W", "Y"),
    "TH": ("F", "S"),
    "DH": ("D", "Z"),
    "S": ("SH", "T"),
    "Z": ("ZH", "D"),
    "SH": ("S", "CH"),
    "CH": ("SH", "T"),
    "JH": ("ZH", "D"),
    "K": ("T", "G"),
    "G": ("D", "K"),
    "T": ("K", "D"),
    "D": ("G", "T"),
    "P": ("B", "F"),
    "B": ("P", "M"),
    "F": ("TH", "P"),
    "V": ("DH", "B"),
    "ER": ("AH", "EH"),
    "IY": ("IH", "EY"),
    "IH": ("IY", "EH"),
    "EH": ("AE", "AH"),
    "AE": ("EH", "AH"),
    "AH": ("AE", "UH"),
    "OW": ("AW", "UH"),
    "UW": ("OW", "UH"),
    "AY": ("EY", "IY"),
    "EY": ("AY", "EH"),
    "OY": ("OW", "UH"),
    "AW": ("OW", "AH"),
    "W": ("Y", "L"),
    "Y": ("W", "IY"),
    "M": ("N", "B"),
    "N": ("M", "NG"),
    "NG": ("N", "G"),
}

MISMATCH_INSERTION_PHONES: Final[tuple[str, ...]] = ("HH", "Y", "W", "AH", "S", "R")

PHONE_TO_IPA: Final[dict[str, str]] = {
    "AE": "\u00e6",
    "AH": "\u028c",
    "AW": "a\u028a",
    "AY": "a\u026a",
    "B": "b",
    "CH": "t\u0283",
    "D": "d",
    "DH": "\u00f0",
    "EH": "\u025b",
    "ER": "\u025d",
    "EY": "e\u026a",
    "F": "f",
    "G": "\u0261",
    "HH": "h",
    "IH": "\u026a",
    "IY": "i",
    "JH": "d\u0292",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "\u014b",
    "OW": "o\u028a",
    "OY": "\u0254\u026a",
    "P": "p",
    "R": "\u0279",
    "S": "s",
    "SH": "\u0283",
    "T": "t",
    "TH": "\u03b8",
    "UH": "\u028a",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "\u0292",
}

PHONE_TO_CLONE_TEXT: Final[dict[str, str]] = {
    "AH": "uh",
    "AE": "a",
    "EH": "eh",
    "IH": "ih",
    "IY": "ee",
    "OW": "oh",
    "UH": "oo",
    "UW": "oo",
    "AY": "eye",
    "EY": "ay",
    "OY": "oy",
    "AW": "ow",
    "ER": "er",
    "R": "r",
    "L": "l",
    "W": "w",
    "Y": "y",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "B": "b",
    "D": "d",
    "G": "g",
    "P": "p",
    "T": "t",
    "K": "k",
    "S": "s",
    "Z": "z",
    "SH": "sh",
    "TH": "th",
    "DH": "dh",
    "F": "f",
    "V": "v",
    "CH": "ch",
    "JH": "j",
    "HH": "h",
    "ZH": "zh",
}
