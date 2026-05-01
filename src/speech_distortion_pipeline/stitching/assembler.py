from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from speech_distortion_pipeline.models import AudioBuffer, AudioSegment


class AudioAssembler(Protocol):
    """Stitches edited source audio and resynthesized fragments into a final waveform."""

    def assemble(
        self,
        original_audio: AudioBuffer,
        edited_audio: AudioBuffer,
        projected_fragments: list[AudioSegment],
    ) -> AudioBuffer:
        raise NotImplementedError


@dataclass
class HeuristicAudioAssembler:
    """Simple overlap-add assembler for bootstrap slices."""

    assembler_name: str = "heuristic_audio_assembler_v1"
    crossfade_ms: int = 12

    def assemble(
        self,
        original_audio: AudioBuffer,
        edited_audio: AudioBuffer,
        projected_fragments: list[AudioSegment],
    ) -> AudioBuffer:
        assembled = list(edited_audio.samples)
        sample_rate_hz = edited_audio.sample_rate_hz
        crossfade = max(1, int(round((self.crossfade_ms / 1000.0) * sample_rate_hz)))

        for fragment in projected_fragments:
            if not fragment.samples:
                continue
            start = max(0, int(round(fragment.start_sec * sample_rate_hz)))
            end = start + len(fragment.samples)
            if end > len(assembled):
                assembled.extend([0.0] * (end - len(assembled)))

            target_span = assembled[start:end]
            mixed = self._mix_fragment(
                base=target_span,
                fragment=list(fragment.samples),
                crossfade=min(crossfade, max(1, len(fragment.samples) // 2)),
            )
            assembled[start:end] = mixed

        return AudioBuffer(
            samples=self._clamp(assembled),
            sample_rate_hz=edited_audio.sample_rate_hz,
            channel_count=edited_audio.channel_count,
            speaker_id=edited_audio.speaker_id,
            metadata={
                **original_audio.metadata,
                **edited_audio.metadata,
                "assembler_name": self.assembler_name,
                "fragment_mix_count": str(len(projected_fragments)),
            },
        )

    def _mix_fragment(self, base: List[float], fragment: List[float], crossfade: int) -> List[float]:
        if not base:
            return fragment
        mixed: List[float] = []
        total = max(len(base), len(fragment))
        if len(base) < total:
            base = base + ([0.0] * (total - len(base)))
        if len(fragment) < total:
            fragment = fragment + ([0.0] * (total - len(fragment)))

        for index in range(total):
            fade = self._fade_weight(index, total, crossfade)
            value = (base[index] * (1.0 - fade)) + (fragment[index] * fade)
            mixed.append(value)
        return mixed

    def _fade_weight(self, index: int, total: int, crossfade: int) -> float:
        if total <= 1:
            return 1.0
        if index < crossfade:
            return float(index + 1) / float(max(crossfade, 1))
        if index >= total - crossfade:
            tail_index = total - index
            return max(0.4, float(tail_index) / float(max(crossfade, 1)))
        return 1.0

    def _clamp(self, samples: List[float]) -> List[float]:
        return [max(-1.0, min(1.0, sample)) for sample in samples]
