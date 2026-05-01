from __future__ import annotations

from typing import Protocol

from speech_distortion_pipeline.models import AudioBuffer


class AudioReader(Protocol):
    def read(self, path: str) -> AudioBuffer:
        raise NotImplementedError


class AudioWriter(Protocol):
    def write(self, path: str, audio: AudioBuffer) -> None:
        raise NotImplementedError


class TranscriptLoader(Protocol):
    def load(self, path: str) -> str:
        raise NotImplementedError
