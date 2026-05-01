from __future__ import annotations

from speech_distortion_pipeline.config import AlignmentConfig

from .base import Aligner
from .runtime import RuntimeCTCAligner


def build_aligner(config: AlignmentConfig) -> Aligner:
    if config.runtime_backend == "torchaudio_ctc":
        return RuntimeCTCAligner(backend_name=config.runtime_backend)
    raise ValueError(f"Unsupported runtime aligner backend: {config.runtime_backend}")
