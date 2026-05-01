from __future__ import annotations

from typing import Protocol

from speech_distortion_pipeline.models import AlignmentResult, AudioBuffer


class Aligner(Protocol):
    """Interface for transcript-constrained alignment backends."""

    def align(self, audio: AudioBuffer, transcript: str) -> AlignmentResult:
        raise NotImplementedError
