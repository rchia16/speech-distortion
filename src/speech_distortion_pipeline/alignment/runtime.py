from __future__ import annotations

import re
from dataclasses import dataclass

from speech_distortion_pipeline.models import AlignmentResult, AudioBuffer
from speech_distortion_pipeline.models.alignment import PhoneSpan, WordSpan


@dataclass
class RuntimeCTCAligner:
    """Runtime aligner wrapper.

    The intended backend is torchaudio CTC forced alignment. Until that is
    integrated, this class emits deterministic word-level spans and one fallback
    phone span per word so downstream modules have stable contracts.
    """

    backend_name: str = "torchaudio_ctc"

    def align(self, audio: AudioBuffer, transcript: str) -> AlignmentResult:
        cleaned = transcript.strip()
        if not cleaned:
            raise ValueError("Transcript must not be empty.")

        words = re.findall(r"[A-Za-z']+", cleaned)
        if not words:
            raise ValueError("Transcript did not contain any alignable words.")

        total_duration_sec = float(len(audio.samples)) / float(audio.sample_rate_hz)
        word_duration_sec = total_duration_sec / float(len(words))

        word_spans: list[WordSpan] = []
        phone_spans: list[PhoneSpan] = []
        for index, word in enumerate(words):
            start_sec = index * word_duration_sec
            end_sec = total_duration_sec if index == len(words) - 1 else (index + 1) * word_duration_sec
            phone = PhoneSpan(
                phone=word.lower(),
                start_sec=start_sec,
                end_sec=end_sec,
                word_index=index,
                confidence=None,
            )
            word_spans.append(
                WordSpan(
                    text=word,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    index=index,
                    phones=[phone],
                )
            )
            phone_spans.append(phone)

        return AlignmentResult(
            transcript=cleaned,
            words=word_spans,
            phones=phone_spans,
            backend_name=f"{self.backend_name}:fallback_uniform",
        )
