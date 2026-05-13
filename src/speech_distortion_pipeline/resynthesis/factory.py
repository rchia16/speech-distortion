from __future__ import annotations

import shutil

from .fragment_synthesizer import (
    CoquiFragmentSynthesizer,
    FragmentSynthesizer,
    HeuristicFragmentSynthesizer,
    HeuristicPronunciationFragmentSynthesizer,
    PronunciationFragmentSynthesizer,
    SayFragmentSynthesizer,
)


def build_fragment_synthesizer() -> FragmentSynthesizer:
    if shutil.which("conda"):
        return CoquiFragmentSynthesizer()
    if shutil.which("say"):
        return SayFragmentSynthesizer()
    return HeuristicFragmentSynthesizer()


def build_pronunciation_fragment_synthesizer() -> PronunciationFragmentSynthesizer:
    return HeuristicPronunciationFragmentSynthesizer()
