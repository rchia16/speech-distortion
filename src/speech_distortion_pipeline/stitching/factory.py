from __future__ import annotations

from .assembler import AudioAssembler, HeuristicAudioAssembler


def build_audio_assembler() -> AudioAssembler:
    return HeuristicAudioAssembler()
