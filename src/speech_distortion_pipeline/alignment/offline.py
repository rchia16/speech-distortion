from __future__ import annotations

from speech_distortion_pipeline.models import AlignmentResult, AudioBuffer


class OfflineReferenceAligner:
    """High-accuracy offline aligner placeholder used for calibration only."""

    def align(self, audio: AudioBuffer, transcript: str) -> AlignmentResult:
        raise NotImplementedError("Offline reference alignment is not implemented yet.")
