from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from speech_distortion_pipeline.models import AudioBuffer, AudioSegment


class TimbreProjector(Protocol):
    """Projects synthetic fragments toward the source speaker identity."""

    def project(self, source_audio: AudioBuffer, fragments: list[AudioSegment]) -> list[AudioSegment]:
        raise NotImplementedError


@dataclass
class HeuristicTimbreProjector:
    """Dependency-free fragment projection using local source envelope cues."""

    projector_name: str = "heuristic_timbre_projector_v1"
    source_mix: float = 0.35

    def project(self, source_audio: AudioBuffer, fragments: list[AudioSegment]) -> list[AudioSegment]:
        projected: List[AudioSegment] = []
        sample_rate_hz = source_audio.sample_rate_hz
        for fragment in fragments:
            if not fragment.samples:
                projected.append(fragment)
                continue

            start = max(0, int(round(fragment.start_sec * sample_rate_hz)))
            end = min(len(source_audio.samples), start + len(fragment.samples))
            source_slice = list(source_audio.samples[start:end])
            if len(source_slice) < len(fragment.samples):
                source_slice.extend([0.0] * (len(fragment.samples) - len(source_slice)))

            projected_samples = self._project_samples(fragment.samples, source_slice)
            projected.append(
                AudioSegment(
                    start_sec=fragment.start_sec,
                    end_sec=fragment.end_sec,
                    samples=projected_samples,
                    label=fragment.label,
                    source=self.projector_name,
                )
            )
        return projected

    def _project_samples(self, fragment: List[float], source: List[float]) -> List[float]:
        frag_rms = self._rms(fragment)
        src_rms = self._rms(source)
        gain = 1.0 if frag_rms <= 1e-6 else src_rms / frag_rms if src_rms > 1e-6 else 0.75
        envelope = self._envelope(source if any(source) else fragment)

        projected: List[float] = []
        for index, sample in enumerate(fragment):
            env = envelope[index] if index < len(envelope) else 1.0
            source_value = source[index] if index < len(source) else 0.0
            value = (sample * gain * (0.65 + (0.35 * env))) + (source_value * self.source_mix)
            projected.append(self._clamp(value))
        return projected

    def _envelope(self, samples: List[float]) -> List[float]:
        if not samples:
            return []
        window = 64
        output: List[float] = []
        for index in range(len(samples)):
            start = max(0, index - window)
            end = min(len(samples), index + window + 1)
            local = samples[start:end]
            mean = sum(abs(value) for value in local) / float(max(len(local), 1))
            output.append(min(1.0, mean * 4.0))
        return output

    def _rms(self, samples: List[float]) -> float:
        if not samples:
            return 0.0
        return (sum(value * value for value in samples) / float(len(samples))) ** 0.5

    def _clamp(self, value: float) -> float:
        return max(-1.0, min(1.0, value))
