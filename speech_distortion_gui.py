"""
Phoneme distances are based on candidates and assessed according to the
granularity set in src/speech_distortion_pipeline/phonology/
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import traceback

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_distortion_pipeline.io import WavAudioReader, WavAudioWriter
from speech_distortion_pipeline.models import AudioBuffer
from speech_distortion_pipeline.models.edits import EditType
from speech_distortion_pipeline.bootstrap import (
    build_hybrid_planning_slice,
    build_hybrid_stitching_slice,
    build_slider_controls,
)
from speech_distortion_pipeline.config import load_hybrid_config
from speech_distortion_pipeline.phonology.blabber_config import (
    BLABBER_VOICE_MODES,
    COQUI_ENV_NAME,
    DEFAULT_GUI_INPUT_FILENAME,
    MISMATCH_INSERTION_PHONES,
    PHONE_SUBSTITUTIONS,
    PHONE_TO_CLONE_TEXT,
    PHONE_TO_IPA,
    normalize_blabber_voice_mode,
    resolve_blabber_voice_profile,
)
from speech_distortion_pipeline.phonology.g2p import HeuristicEnglishG2P
from speech_distortion_pipeline.phonology.phone_distance_reference import (
    bucketed_phone_choice,
    bucketed_phone_sequence,
    global_blabber_preset_index,
    resolve_per_phoneme_blabber_sequences,
    resolve_phone_blabber_sequence,
    resolve_soft_global_blabber_sequences,
    ranked_neighbors_from_reference,
)
from speech_distortion_pipeline.resynthesis.fragment_synthesizer import (
    CoquiFragmentSynthesizer,
    CoquiSynthesisError,
    resolve_python_command,
)
from speech_distortion_pipeline.resynthesis.phoneme_render_backend import (
    DEFAULT_KOKORO_VOICE,
    KOKORO_VOICE_OPTIONS,
    PHONEME_RENDER_BACKENDS,
    PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    PHONEME_RENDER_BACKEND_KOKORO,
    PhonemeRenderError,
    backend_metadata,
    default_woman_phoneme_render_backend,
    render_with_phoneme_backend,
)
from speech_distortion_pipeline.resynthesis.style_transfer import (
    DEFAULT_STYLE_TRANSFER_PRESETS_PATH,
    STYLE_TRANSFER_BACKENDS,
    STYLE_TRANSFER_BACKEND_NONE,
    apply_style_transfer,
    male_style_transfer_presets,
)
DEFAULT_INPUT = ROOT / DEFAULT_GUI_INPUT_FILENAME
DEFAULT_PRONUNCIATION_SLIDER_CONFIG = ROOT / "hybrid.yaml"
MAX_DROPOUT_FRACTION = 0.8
MAX_DROPOUT_SILENCE_MS = 200.0
GLOBAL_BLABBER_MAX_PHONEME_DISTANCE_VALUES = tuple(f"{value / 10.0:.1f}" for value in range(8, 101))
DEFAULT_GLOBAL_BLABBER_MAX_PHONEME_DISTANCE = "4.0"
DEFAULT_MALE_COQUI_SPEAKER = "p226"
MALE_COQUI_SPEAKER_OPTIONS = (
    "p226",
    "p227",
    "p232",
    "p243",
    "p256",
    "p270",
    "p287",
)
DEFAULT_STYLE_TRANSFER_BACKEND = STYLE_TRANSFER_BACKEND_NONE
DEFAULT_PHONEME_RENDER_BACKEND = default_woman_phoneme_render_backend()


PHONE_CLASS_WEIGHTS: dict[str, float] = {
    "AE": 1.9,
    "AH": 1.7,
    "AW": 1.9,
    "AY": 1.9,
    "EH": 1.7,
    "ER": 1.8,
    "EY": 1.8,
    "IH": 1.6,
    "IY": 1.8,
    "OW": 1.8,
    "OY": 1.9,
    "UH": 1.6,
    "UW": 1.8,
    "R": 1.2,
    "L": 1.2,
    "W": 1.1,
    "Y": 1.1,
    "M": 1.2,
    "N": 1.2,
    "NG": 1.2,
    "S": 1.0,
    "Z": 1.0,
    "SH": 1.0,
    "ZH": 1.0,
    "TH": 1.0,
    "DH": 1.0,
    "F": 1.0,
    "V": 1.0,
    "HH": 0.9,
    "CH": 1.0,
    "JH": 1.0,
    "P": 0.85,
    "B": 0.9,
    "T": 0.85,
    "D": 0.9,
    "K": 0.85,
    "G": 0.9,
}


def clamp(samples: np.ndarray) -> np.ndarray:
    return np.clip(samples, -1.0, 1.0)


def lerp(start: float, end: float, amount: float) -> float:
    return start + ((end - start) * amount)


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def to_numpy(audio: AudioBuffer) -> np.ndarray:
    return np.asarray(audio.samples, dtype=np.float32)


def from_numpy(samples: np.ndarray, template: AudioBuffer, **metadata: str) -> AudioBuffer:
    merged = dict(template.metadata)
    merged.update(metadata)
    return AudioBuffer(
        samples=clamp(samples).astype(np.float32).tolist(),
        sample_rate_hz=template.sample_rate_hz,
        channel_count=template.channel_count,
        speaker_id=template.speaker_id,
        metadata=merged,
    )


def volume_dropout_fraction_from_quality(quality: float) -> float:
    return clamp_unit(1.0 - clamp_unit(quality)) * MAX_DROPOUT_FRACTION


def slider_caption_for_mode(mode: str) -> str:
    if mode == "volume_dropout":
        return "Dropout"
    return "Quality"


def slider_value_text_for_mode(mode: str, quality: float) -> str:
    if mode == "volume_dropout":
        dropped_percent = int(round(volume_dropout_fraction_from_quality(quality) * 100.0))
        return f"{dropped_percent}%"
    return f"{clamp_unit(quality):.2f}"


def subtitle_text_for_mode(mode: str) -> str:
    if mode == "volume_dropout":
        return (
            "Volume dropout inserts silence gaps in place. Use Breakpoints='auto' to apply dropout at inferred "
            "phoneme segments from the transcript."
        )
    if mode == "phone_style":
        return (
            "Phone style mixes the clean signal with white or pink noise and can vary over time "
            "using breakpoint envelopes like 0.0:0.0,0.5:1.0,1.0:0.0, or Breakpoints='auto' for phoneme segments."
        )
    if mode == "blabber":
        return (
            "Blabber synthesizes a mutated phone sequence. Breakpoints can be manual or 'auto' to crossfade by "
            "inferred phoneme segments."
        )
    if mode == "pronunciation_safe":
        return (
            "Pronunciation-safe sliders keep the target word recognizable while independently controlling "
            "jibberish drift, acoustic clarity loss, and timing instability."
        )
    return "One augmentation at a time. Quality 1.0 keeps the original signal; 0.0 is worst quality."


def resample_array(samples: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if samples.size == target_length:
        return samples.astype(np.float32, copy=True)
    if samples.size == 1:
        return np.full(target_length, float(samples[0]), dtype=np.float32)

    src_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
    dst_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


def parse_breakpoints(breakpoints_str: str) -> List[Tuple[float, float]]:
    breakpoints: List[Tuple[float, float]] = []
    if not breakpoints_str.strip():
        return [(0.0, 1.0), (1.0, 1.0)]
    for pair in breakpoints_str.split(","):
        try:
            pos_str, inten_str = pair.split(":")
            position = float(pos_str.strip())
            intensity = float(inten_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Breakpoint '{pair}' is invalid. Use 'position:intensity'."
            ) from exc
        if not (0.0 <= position <= 1.0):
            raise ValueError(f"Breakpoint position {position} is outside [0.0, 1.0].")
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"Breakpoint intensity {intensity} is outside [0.0, 1.0].")
        breakpoints.append((position, intensity))
    breakpoints.sort(key=lambda item: item[0])
    if breakpoints[0][0] > 0.0:
        breakpoints.insert(0, (0.0, breakpoints[0][1]))
    if breakpoints[-1][0] < 1.0:
        breakpoints.append((1.0, breakpoints[-1][1]))
    return breakpoints


def build_envelope(num_samples: int, breakpoints: List[Tuple[float, float]]) -> np.ndarray:
    positions = [int(position * max(num_samples - 1, 1)) for position, _ in breakpoints]
    intensities = [intensity for _, intensity in breakpoints]
    return np.interp(np.arange(num_samples), positions, intensities).astype(np.float32)


def build_phone_segments(phones: Sequence[str]) -> List[Tuple[float, float, str]]:
    if not phones:
        return []
    weights = [PHONE_CLASS_WEIGHTS.get(phone, 1.0) for phone in phones]
    total_weight = sum(weights) or 1.0
    segments: List[Tuple[float, float, str]] = []
    cursor = 0.0
    for index, phone in enumerate(phones):
        width = weights[index] / total_weight
        start = cursor
        end = 1.0 if index == len(phones) - 1 else min(1.0, cursor + width)
        segments.append((start, end, phone))
        cursor = end
    return segments


def phone_segments_to_sample_spans(
    phones: Sequence[str],
    total_samples: int,
) -> List[Tuple[int, int, str]]:
    if total_samples <= 0:
        return []
    relative = build_phone_segments(phones)
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for index, (start_ratio, end_ratio, phone) in enumerate(relative):
        start = cursor if index > 0 else 0
        end = total_samples if index == len(relative) - 1 else max(start + 1, int(round(end_ratio * total_samples)))
        spans.append((start, min(total_samples, end), phone))
        cursor = min(total_samples, end)
    return spans


def _detect_activity_window(samples: np.ndarray, sample_rate_hz: int) -> tuple[int, int]:
    if samples.size == 0:
        return 0, 0
    envelope_window = max(1, int(round(sample_rate_hz * 0.01)))
    kernel = np.ones(envelope_window, dtype=np.float32) / float(envelope_window)
    envelope = np.convolve(np.abs(samples), kernel, mode="same")
    peak = float(np.max(envelope))
    if peak <= 1e-8:
        return 0, samples.size
    threshold = max(peak * 0.08, 1e-4)
    active = np.flatnonzero(envelope >= threshold)
    if active.size == 0:
        return 0, samples.size
    pad = max(1, int(round(sample_rate_hz * 0.01)))
    start = max(0, int(active[0]) - pad)
    end = min(samples.size, int(active[-1]) + pad + 1)
    if end <= start:
        return 0, samples.size
    return start, end


def _select_boundary_indices(scores: np.ndarray, count: int, min_gap: int) -> List[int]:
    if count <= 0 or scores.size == 0:
        return []
    ranked = list(np.argsort(scores)[::-1])
    selected: List[int] = []
    for candidate in ranked:
        if scores[candidate] <= 0.0:
            continue
        if any(abs(candidate - existing) < min_gap for existing in selected):
            continue
        selected.append(int(candidate))
        if len(selected) >= count:
            break
    return sorted(selected)


def _audio_segment_boundaries(
    samples: np.ndarray,
    sample_rate_hz: int,
    segment_count: int,
) -> List[int]:
    if segment_count <= 1 or samples.size <= 1:
        return [0, max(1, samples.size)]

    active_start, active_end = _detect_activity_window(samples, sample_rate_hz)
    active_length = max(1, active_end - active_start)
    frame_length = min(max(64, int(round(sample_rate_hz * 0.025))), max(64, active_length))
    hop_length = max(32, int(round(sample_rate_hz * 0.005)))
    if active_length <= frame_length:
        return [0] + [
            int(round(samples.size * index / float(segment_count)))
            for index in range(1, segment_count)
        ] + [samples.size]

    window = np.hanning(frame_length).astype(np.float32)
    frame_starts = list(range(active_start, max(active_start + 1, active_end - frame_length + 1), hop_length))
    if not frame_starts:
        return [0] + [
            int(round(samples.size * index / float(segment_count)))
            for index in range(1, segment_count)
        ] + [samples.size]

    flux_values: list[float] = []
    rms_values: list[float] = []
    previous_magnitude: np.ndarray | None = None
    for frame_start in frame_starts:
        frame = samples[frame_start : frame_start + frame_length]
        if frame.shape[0] < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.shape[0]))
        spectrum = np.abs(np.fft.rfft(frame * window))
        if previous_magnitude is None:
            flux_values.append(0.0)
        else:
            flux_values.append(float(np.sum(np.maximum(0.0, spectrum - previous_magnitude))))
        previous_magnitude = spectrum
        rms_values.append(float(np.sqrt(np.mean(np.square(frame))) + 1e-8))

    flux = np.asarray(flux_values, dtype=np.float32)
    rms = np.asarray(rms_values, dtype=np.float32)
    if flux.size == 0:
        return [0] + [
            int(round(samples.size * index / float(segment_count)))
            for index in range(1, segment_count)
        ] + [samples.size]
    flux = flux / float(np.max(flux) + 1e-8)
    rms_delta = np.abs(np.diff(rms, prepend=rms[:1]))
    rms_delta = rms_delta / float(np.max(rms_delta) + 1e-8)
    novelty = (0.8 * flux) + (0.2 * rms_delta)
    if novelty.size >= 3:
        novelty = np.convolve(novelty, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")

    min_gap = max(1, int(round(len(frame_starts) / float(segment_count * 2))))
    chosen = _select_boundary_indices(novelty[1:-1], segment_count - 1, min_gap)
    chosen = [index + 1 for index in chosen]

    while len(chosen) < segment_count - 1:
        existing_positions = [active_start] + [
            min(active_end - 1, frame_starts[index] + (frame_length // 2))
            for index in chosen
        ] + [active_end]
        existing_positions.sort()
        largest_gap_index = max(
            range(len(existing_positions) - 1),
            key=lambda idx: existing_positions[idx + 1] - existing_positions[idx],
        )
        midpoint = int(round((existing_positions[largest_gap_index] + existing_positions[largest_gap_index + 1]) * 0.5))
        nearest_index = min(
            range(len(frame_starts)),
            key=lambda idx: abs((frame_starts[idx] + (frame_length // 2)) - midpoint),
        )
        if nearest_index not in chosen and 0 < nearest_index < len(frame_starts) - 1:
            chosen.append(nearest_index)
            chosen.sort()
        else:
            break

    boundaries = [0]
    for frame_index in chosen[: segment_count - 1]:
        center = frame_starts[frame_index] + (frame_length // 2)
        boundaries.append(int(min(samples.size - 1, max(1, center))))
    boundaries.append(samples.size)
    boundaries = sorted(boundaries)

    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            boundaries[index] = min(samples.size, boundaries[index - 1] + 1)
    boundaries[-1] = samples.size
    boundaries[0] = 0
    return boundaries


def derive_phone_sample_spans(
    audio: AudioBuffer,
    transcript: str,
    phones: Sequence[str] | None = None,
) -> List[Tuple[int, int, str]]:
    cleaned = transcript.strip()
    if not cleaned:
        return []
    g2p = HeuristicEnglishG2P()
    resolved_phones = list(phones) if phones is not None else g2p.phonemize_word(cleaned)
    if not resolved_phones:
        return []

    total_samples = len(audio.samples)
    if total_samples <= 0:
        return []

    boundaries = _audio_segment_boundaries(
        to_numpy(audio),
        audio.sample_rate_hz,
        len(resolved_phones),
    )
    spans: List[Tuple[int, int, str]] = []
    for index, phone in enumerate(resolved_phones):
        start = boundaries[index]
        end = boundaries[index + 1] if index + 1 < len(boundaries) else total_samples
        spans.append((start, max(start + 1, min(total_samples, end)), phone))
    if spans:
        spans[-1] = (spans[-1][0], total_samples, spans[-1][2])
    return spans


def build_breakpoints_from_segments(
    segment_values: Sequence[Tuple[float, float, float]],
) -> List[Tuple[float, float]]:
    if not segment_values:
        return [(0.0, 1.0), (1.0, 1.0)]
    breakpoints: List[Tuple[float, float]] = []
    for index, (start, end, value) in enumerate(segment_values):
        intensity = clamp_unit(value)
        if index == 0:
            breakpoints.append((start, intensity))
        elif start > breakpoints[-1][0]:
            breakpoints.append((start, intensity))
        else:
            breakpoints[-1] = (breakpoints[-1][0], intensity)
        breakpoints.append((end, intensity))
    if breakpoints[0][0] > 0.0:
        breakpoints.insert(0, (0.0, breakpoints[0][1]))
    if breakpoints[-1][0] < 1.0:
        breakpoints.append((1.0, breakpoints[-1][1]))
    return breakpoints


def format_breakpoints(breakpoints: Sequence[Tuple[float, float]]) -> str:
    return ",".join(f"{position:.3f}:{intensity:.3f}" for position, intensity in breakpoints)


def phone_segment_intensity(phone: str, base: float, mode: str) -> float:
    if phone in {"S", "Z", "SH", "ZH", "TH", "DH", "F", "V", "CH", "JH"}:
        return clamp_unit(base + 0.18)
    if phone in {"P", "B", "T", "D", "K", "G"}:
        return clamp_unit(base + 0.1)
    if phone in {"AE", "AH", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}:
        return clamp_unit(base - (0.12 if mode != "blabber" else 0.05))
    return clamp_unit(base)


def resolve_breakpoints(
    breakpoints_str: str,
    audio: AudioBuffer | None,
    transcript: str,
    mode: str,
    quality: float,
    source_phones: Sequence[str] | None = None,
    mutated_phones: Sequence[str] | None = None,
    phoneme_qualities: Sequence[float] | None = None,
) -> tuple[List[Tuple[float, float]], str]:
    spec = breakpoints_str.strip()
    if spec and spec.lower() != "auto":
        parsed = parse_breakpoints(spec)
        return parsed, format_breakpoints(parsed)

    cleaned = transcript.strip()
    if not cleaned:
        parsed = parse_breakpoints("0.0:1.0,1.0:1.0")
        return parsed, format_breakpoints(parsed)

    g2p = HeuristicEnglishG2P()
    phones = list(source_phones) if source_phones else g2p.phonemize_word(cleaned)
    if not phones:
        parsed = parse_breakpoints("0.0:1.0,1.0:1.0")
        return parsed, format_breakpoints(parsed)

    severity = clamp_unit(1.0 - quality)
    if audio is not None:
        sample_spans = derive_phone_sample_spans(audio, cleaned, phones)
        total_samples = max(1, len(audio.samples))
        segments = [
            (
                start / float(total_samples),
                end / float(total_samples),
                phone,
            )
            for start, end, phone in sample_spans
        ]
    else:
        segments = build_phone_segments(phones)
    segment_values: List[Tuple[float, float, float]] = []
    if mode == "blabber" and mutated_phones is not None:
        for index, (start, end, phone) in enumerate(segments):
            mutated = index >= len(mutated_phones) or mutated_phones[index] != phone
            value = clamp_unit((severity * 0.2) + (severity * 0.95 if mutated else 0.0))
            if phoneme_qualities is not None and index < len(phoneme_qualities):
                local_intensity = 1.0 - clamp_unit(float(phoneme_qualities[index]))
                value = clamp_unit((value + local_intensity) * 0.5)
            segment_values.append((start, end, value))
        if len(mutated_phones) > len(phones) and segment_values:
            start, end, value = segment_values[-1]
            segment_values[-1] = (start, end, max(value, severity))
    else:
        for index, (start, end, phone) in enumerate(segments):
            value = phone_segment_intensity(phone, severity, mode)
            if phoneme_qualities is not None and index < len(phoneme_qualities):
                local_intensity = 1.0 - clamp_unit(float(phoneme_qualities[index]))
                value = clamp_unit((value + local_intensity) * 0.5)
            segment_values.append((start, end, value))

    parsed = build_breakpoints_from_segments(segment_values)
    return parsed, format_breakpoints(parsed)


def generate_pink_noise(n_samples: int, sample_rate: int) -> np.ndarray:
    white = np.random.randn(n_samples)
    white_fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink_fft = white_fft * scale
    pink = np.fft.irfft(pink_fft, n=n_samples)
    max_abs = np.max(np.abs(pink))
    if max_abs > 0:
        pink /= max_abs
    return pink.astype(np.float32)


def generate_noise(n_samples: int, sample_rate: int, noise_type: str) -> np.ndarray:
    if noise_type == "white":
        noise = np.random.randn(n_samples).astype(np.float32)
        peak = float(np.max(np.abs(noise)) + 1e-8)
        return noise / peak
    if noise_type == "pink":
        return generate_pink_noise(n_samples, sample_rate)
    raise ValueError(f"Unsupported noise type '{noise_type}'.")


def apply_time_envelope(
    clean: np.ndarray,
    effected: np.ndarray,
    breakpoints_str: str,
) -> np.ndarray:
    if clean.shape[0] != effected.shape[0]:
        effected = resample_array(effected, clean.shape[0])
    breakpoints = parse_breakpoints(breakpoints_str)
    envelope = build_envelope(clean.shape[0], breakpoints)
    return clamp((clean * (1.0 - envelope)) + (effected * envelope))


def apply_time_envelope_breakpoints(
    clean: np.ndarray,
    effected: np.ndarray,
    breakpoints: Sequence[Tuple[float, float]],
) -> np.ndarray:
    if clean.shape[0] != effected.shape[0]:
        effected = resample_array(effected, clean.shape[0])
    envelope = build_envelope(clean.shape[0], breakpoints)
    return clamp((clean * (1.0 - envelope)) + (effected * envelope))


def apply_phone_style_augmentation(
    audio: AudioBuffer,
    quality: float,
    noise_type: str,
    breakpoints_str: str,
    transcript: str = "",
    phoneme_qualities: Sequence[float] | None = None,
) -> AudioBuffer:
    signal = to_numpy(audio)
    noise = generate_noise(signal.shape[0], audio.sample_rate_hz, noise_type)
    signal_rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-8)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))) + 1e-8)
    matched_noise = noise * (signal_rms / noise_rms)
    severity = 1.0 - quality
    fully_effected = clamp((signal * quality) + (matched_noise * severity))
    breakpoints, breakpoint_label = resolve_breakpoints(
        breakpoints_str,
        audio,
        transcript,
        "phone_style",
        quality,
        phoneme_qualities=phoneme_qualities,
    )
    output = apply_time_envelope_breakpoints(signal, fully_effected, breakpoints)
    return from_numpy(
        output,
        audio,
        augmentation="phone_style",
        quality=f"{quality:.3f}",
        noise_type=noise_type,
        breakpoints=breakpoint_label,
    )


def make_audio_segment(audio: AudioBuffer, samples: np.ndarray) -> AudioBuffer:
    return AudioBuffer(
        samples=samples.astype(np.float32).tolist(),
        sample_rate_hz=audio.sample_rate_hz,
        channel_count=audio.channel_count,
        speaker_id=audio.speaker_id,
        metadata=dict(audio.metadata),
    )


def augment_by_phoneme_segments(
    audio: AudioBuffer,
    transcript: str,
    phoneme_qualities: Sequence[float],
    mode: str,
    noise_type: str = "white",
) -> AudioBuffer:
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Transcript is required for per-phoneme augmentation.")

    g2p = HeuristicEnglishG2P()
    phones = g2p.phonemize_word(cleaned)
    if not phones:
        raise ValueError("Could not derive phones from the transcript.")

    signal = to_numpy(audio)
    spans = derive_phone_sample_spans(audio, cleaned, phones)
    output = signal.copy()

    for index, (start, end, _phone) in enumerate(spans):
        local_quality = clamp_unit(float(phoneme_qualities[index])) if index < len(phoneme_qualities) else 1.0
        segment_audio = make_audio_segment(audio, signal[start:end])
        if mode == "static_noise":
            effected = apply_static_noise(segment_audio, local_quality)
        elif mode == "volume_dropout":
            effected = apply_volume_dropout(segment_audio, local_quality)
        elif mode == "phone_style":
            effected = apply_phone_style_augmentation(
                segment_audio,
                local_quality,
                noise_type,
                "0.0:1.0,1.0:1.0",
                transcript="",
                phoneme_qualities=None,
            )
        else:
            raise ValueError(f"Unsupported per-phoneme augmentation mode '{mode}'.")
        output[start:end] = to_numpy(effected)[: max(0, end - start)]

    metadata = {
        "augmentation": mode,
        "transcript": cleaned,
        "source_phones": "-".join(phones),
        "phoneme_qualities": ",".join(f"{clamp_unit(float(v)):.2f}" for v in phoneme_qualities[: len(phones)]),
        "segmentation": "audio",
    }
    if mode == "phone_style":
        metadata["noise_type"] = noise_type
    return from_numpy(output, audio, **metadata)


def apply_static_noise(audio: AudioBuffer, quality: float) -> AudioBuffer:
    signal = to_numpy(audio)
    rng = np.random.default_rng(17)
    signal_rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-8)
    noise = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32)
    colored = np.empty_like(noise)
    previous = 0.0
    alpha = 0.72
    for index, sample in enumerate(noise):
        previous = (alpha * previous) + ((1.0 - alpha) * float(sample))
        colored[index] = previous

    colored_rms = float(np.sqrt(np.mean(np.square(colored))) + 1e-8)
    matched_noise = colored * (signal_rms / colored_rms)
    mix = (signal * quality) + (matched_noise * (1.0 - quality))
    return from_numpy(mix, audio, augmentation="static_noise", quality=f"{quality:.3f}")


def apply_volume_dropout(audio: AudioBuffer, quality: float) -> AudioBuffer:
    quality = clamp_unit(quality)
    drop_fraction = volume_dropout_fraction_from_quality(quality)
    if drop_fraction <= 0.0001:
        return from_numpy(
            to_numpy(audio),
            audio,
            augmentation="volume_dropout",
            quality=f"{quality:.3f}",
            drop_fraction=f"{drop_fraction:.3f}",
            drop_percent=f"{drop_fraction * 100.0:.1f}",
        )

    signal = to_numpy(audio)
    if signal.size == 0:
        return from_numpy(
            signal,
            audio,
            augmentation="volume_dropout",
            quality=f"{quality:.3f}",
            drop_fraction=f"{drop_fraction:.3f}",
            drop_percent=f"{drop_fraction * 100.0:.1f}",
        )

    rng = random.Random(31)
    sample_rate = audio.sample_rate_hz
    normalized = drop_fraction / MAX_DROPOUT_FRACTION
    frame_samples = max(1, int(round(sample_rate * 0.01)))
    frame_count = max(1, int(math.ceil(signal.shape[0] / float(frame_samples))))
    target_drop_frames = min(frame_count, max(0, int(round(frame_count * drop_fraction))))

    dropped_frames = np.zeros(frame_count, dtype=bool)
    remaining = target_drop_frames
    max_silence_frames = max(1, int(math.floor((MAX_DROPOUT_SILENCE_MS / 1000.0) * sample_rate / frame_samples)))
    max_burst_frames = max(1, min(max_silence_frames, int(round(lerp(1.0, float(max_silence_frames), normalized)))))
    while remaining > 0:
        start = rng.randrange(frame_count)
        burst = min(remaining, rng.randint(1, max_burst_frames))
        for frame_index in range(start, min(frame_count, start + burst)):
            if dropped_frames[frame_index]:
                continue
            dropped_frames[frame_index] = True
            remaining -= 1
            if remaining <= 0:
                break

    keep_mask = np.ones(signal.shape[0], dtype=np.float32)
    for frame_index, is_dropped in enumerate(dropped_frames):
        if not is_dropped:
            continue
        start = frame_index * frame_samples
        end = min(signal.shape[0], start + frame_samples)
        keep_mask[start:end] = 0.0

    dropped_sample_fraction = 1.0 - (float(np.count_nonzero(keep_mask)) / float(signal.shape[0]))

    return from_numpy(
        signal * keep_mask,
        audio,
        augmentation="volume_dropout",
        quality=f"{quality:.3f}",
        drop_fraction=f"{dropped_sample_fraction:.3f}",
        drop_percent=f"{dropped_sample_fraction * 100.0:.1f}",
    )


def pick_substituted_phones(
    phones: Sequence[str],
    severity: float,
    phoneme_qualities: Sequence[float] | None = None,
) -> list[str]:
    substituted: list[str] = []
    for index, phone in enumerate(phones):
        local_quality = 1.0 - clamp_unit(severity)
        if phoneme_qualities is not None and index < len(phoneme_qualities):
            local_quality = clamp_unit(float(phoneme_qualities[index]))
        substituted.extend(build_phone_mismatch_sequence(phone, local_quality))
    return substituted


def phones_to_ipa(phones: Sequence[str]) -> str:
    ipa = []
    for phone in phones:
        symbol = PHONE_TO_IPA.get(phone)
        if symbol is None:
            raise ValueError(f"No IPA mapping defined for phone '{phone}'.")
        ipa.append(symbol)
    return "".join(ipa)


def phones_to_clone_text(phones: Sequence[str]) -> str:
    try:
        return "".join(PHONE_TO_CLONE_TEXT[phone] for phone in phones)
    except KeyError as exc:
        raise ValueError(f"No clone-text mapping defined for phone '{exc.args[0]}'.") from exc


def render_blabber_sequence(
    audio: AudioBuffer,
    phones: Sequence[str],
    voice_mode: str,
    speaker_name: str | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
) -> tuple[np.ndarray, str, dict[str, str]]:
    profile = resolve_blabber_voice_profile(voice_mode)
    if (
        profile.uses_source_speaker_wav
        or speaker_name
        or not profile.prephonemized
        or profile.input_encoding != "ipa"
    ):
        synthesizer = CoquiFragmentSynthesizer(
            conda_env_name=COQUI_ENV_NAME,
            model_name=profile.model_name,
            speaker_wav_path=resolve_source_speaker_wav_path(audio, True) if profile.uses_source_speaker_wav else None,
            speaker_name=speaker_name,
        )
        render_text = phones_to_clone_text(phones) if profile.input_encoding == "clone_text" else phones_to_ipa(phones)
        rendered = synthesizer._render_with_coqui(
            render_text,
            audio.sample_rate_hz,
            prephonemized=profile.prephonemized,
            language=profile.language,
        )
        if not rendered:
            raise RuntimeError(f"Coqui synthesis produced no audio in conda env '{COQUI_ENV_NAME}'.")
        return np.asarray(rendered, dtype=np.float32), profile.model_name, {
            "phoneme_render_backend": "coqui_voice_profile_direct",
            "phoneme_render_model_name": profile.model_name,
            "phoneme_render_input_encoding": profile.input_encoding,
            "phoneme_render_prephonemized": "1" if profile.prephonemized else "0",
        }

    try:
        result = render_with_phoneme_backend(
            audio,
            phones,
            backend=phoneme_render_backend,
            conda_env_name=COQUI_ENV_NAME,
            backend_voice=backend_voice,
        )
    except PhonemeRenderError as exc:
        raise RuntimeError(f"Phoneme render backend failed in conda env '{COQUI_ENV_NAME}'.\n\n{exc}") from exc
    return np.asarray(result.samples, dtype=np.float32), result.model_name, backend_metadata(result)


def mutate_phone(phone: str, severity: float) -> str:
    quality = 1.0 - clamp_unit(severity)
    return bucketed_phone_choice(
        phone,
        quality,
        fallback_choices=PHONE_SUBSTITUTIONS.get(phone, ()),
    )


def phone_mismatch_candidates(phone: str) -> list[str]:
    ranked = ranked_neighbors_from_reference(phone)[:4]
    if ranked:
        return ranked
    # Fallback only
    return list(PHONE_SUBSTITUTIONS.get(phone, ()))


def build_phone_mismatch_sequence(phone: str, quality: float) -> list[str]:
    sequence, _preset_index, _distance = resolve_phone_blabber_sequence(
        phone,
        clamp_unit(quality),
        fallback_choices=PHONE_SUBSTITUTIONS.get(phone, ()),
    )
    return sequence


def build_per_phoneme_blabber_sequence(
    source_phones: Sequence[str],
    phoneme_qualities: Sequence[float],
) -> tuple[list[str], list[int], list[float], float, bool, list[list[str]], bool]:
    candidate_sequences, preset_indices, distances, total_distance, \
            expansion_used, numeric_entries_used = \
            resolve_per_phoneme_blabber_sequences(
                source_phones,
                [clamp_unit(float(value)) for value in phoneme_qualities],
                fallback_map=PHONE_SUBSTITUTIONS,
            )
    mutated = [phone for sequence in candidate_sequences for phone in sequence]
    return mutated, preset_indices, distances, total_distance, \
            expansion_used, candidate_sequences, numeric_entries_used


def resolve_blabber_phone_sequence(
    transcript: str,
    quality: float,
    phoneme_qualities: Sequence[float] | None = None,
    max_phoneme_distance: float | None = None,
) -> tuple[
    str,
    list[str],
    list[str],
    int | None,
    bool,
    float | None,
    list[int] | None,
    list[float] | None,
    list[list[str]] | None,
    bool | None,
]:
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Blabber mode requires a single-word transcript.")
    if len(cleaned.split()) != 1:
        raise ValueError("Blabber mode currently supports exactly one word.")

    g2p = HeuristicEnglishG2P()
    source_phones = g2p.phonemize_word(cleaned)
    if not source_phones:
        raise ValueError("Could not derive phones from the transcript.")

    if phoneme_qualities is not None:
        (
            mutated_phones,
            per_phone_preset_indices,
            per_phone_distances,
            total_distance,
            expansion_used,
            per_phone_sequences,
            per_phone_numeric_entries_used,
        ) = build_per_phoneme_blabber_sequence(source_phones, phoneme_qualities)
        preset_index = None
    else:
        severity = 1.0 - quality
        if severity <= 0.001:
            mutated_phones = list(source_phones)
            preset_index = global_blabber_preset_index(quality)
            expansion_used = False
            total_distance = 0.0
            per_phone_preset_indices = None
            per_phone_distances = None
            per_phone_sequences = None
            per_phone_numeric_entries_used = None
        else:
            candidate_sequences, preset_index, expansion_used, total_distance = resolve_soft_global_blabber_sequences(
                source_phones,
                quality,
                fallback_map=PHONE_SUBSTITUTIONS,
                max_phoneme_distance=max_phoneme_distance,
            )
            mutated_phones = [phone for sequence in candidate_sequences for phone in sequence]
            per_phone_preset_indices = None
            per_phone_distances = None
            per_phone_sequences = None
            per_phone_numeric_entries_used = None
    return (
        cleaned,
        source_phones,
        mutated_phones,
        preset_index,
        expansion_used,
        total_distance,
        per_phone_preset_indices,
        per_phone_distances,
        per_phone_sequences,
        per_phone_numeric_entries_used,
    )


def render_blabber_from_resolved_phones(
    audio: AudioBuffer,
    transcript: str,
    source_phones: Sequence[str],
    mutated_phones: Sequence[str],
    voice_mode: str = "woman",
    speaker_name: str | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
    quality: float | None = None,
    phoneme_qualities: Sequence[float] | None = None,
    segmentation: str | None = None,
    breakpoints: str | None = None,
    global_preset_index: int | None = None,
    expansion_used: bool | None = None,
    global_distance_total: float | None = None,
    global_max_phoneme_distance: float | None = None,
    per_phone_preset_indices: Sequence[int] | None = None,
    per_phone_distances: Sequence[float] | None = None,
    per_phone_sequences: Sequence[Sequence[str]] | None = None,
    per_phone_numeric_entries_used: bool | None = None,
) -> AudioBuffer:
    if not mutated_phones:
        raise ValueError("Blabber sequence is empty.")

    try:
        normalized_voice_mode = normalize_blabber_voice_mode(voice_mode)
        resolved_speaker_name = speaker_name.strip() if speaker_name else None
        if normalized_voice_mode != "man":
            resolved_speaker_name = None
        output, coqui_model_name, render_backend_meta = render_blabber_sequence(
            audio,
            mutated_phones,
            normalized_voice_mode,
            speaker_name=resolved_speaker_name,
            phoneme_render_backend=phoneme_render_backend,
            backend_voice=backend_voice,
        )
    except CoquiSynthesisError as exc:
        raise RuntimeError(f"Coqui synthesis failed in conda env '{COQUI_ENV_NAME}'.\n\n{exc}") from exc

    metadata: dict[str, str] = {
        "augmentation": "blabber",
        "transcript": transcript,
        "source_phones": "-".join(source_phones),
        "substituted_phones": "-".join(mutated_phones),
        "substituted_ipa": phones_to_ipa(mutated_phones),
        "mutation_count": str(
            sum(1 for src, dst in zip(source_phones, mutated_phones[: len(source_phones)]) if src != dst)
            + max(0, len(mutated_phones) - len(source_phones))
        ),
        "voice_mode": normalized_voice_mode,
        "coqui_model_name": coqui_model_name,
        "source_speaker_wav": "1" if normalized_voice_mode == "source_clone" else "0",
    }
    metadata.update(render_backend_meta)
    if resolved_speaker_name:
        metadata["coqui_speaker"] = resolved_speaker_name
    if quality is not None:
        metadata["quality"] = f"{quality:.3f}"
    if phoneme_qualities is not None:
        metadata["phoneme_qualities"] = ",".join(
            f"{clamp_unit(float(v)):.2f}" for v in phoneme_qualities[: len(source_phones)]
        )
    if segmentation:
        metadata["segmentation"] = segmentation
    if breakpoints:
        metadata["breakpoints"] = breakpoints
    if global_preset_index is not None:
        metadata["global_preset_index"] = str(global_preset_index)
    if expansion_used is not None:
        metadata["expansion_used"] = "1" if expansion_used else "0"
    if global_distance_total is not None:
        metadata["global_distance_total"] = f"{global_distance_total:.6f}"
    if global_max_phoneme_distance is not None:
        metadata["global_max_phoneme_distance"] = f"{global_max_phoneme_distance:.3f}"
    if per_phone_preset_indices is not None:
        metadata["per_phone_preset_indices"] = ",".join(str(int(value)) for value in per_phone_preset_indices)
    if per_phone_distances is not None:
        whole_word_distance_total = sum(float(value) for value in per_phone_distances)
        metadata["per_phone_distances"] = ",".join(f"{float(value):.6f}" for value in per_phone_distances)
        metadata["per_phone_distance_total"] = f"{whole_word_distance_total:.6f}"
        metadata["whole_word_distance_total"] = f"{whole_word_distance_total:.6f}"
    if per_phone_sequences is not None:
        metadata["per_phone_sequences"] = ";".join("-".join(sequence) for sequence in per_phone_sequences)
    if per_phone_numeric_entries_used is not None:
        metadata["per_phone_numeric_entries_used"] = "1" if per_phone_numeric_entries_used else "0"
        metadata["per_phone_reference_warning"] = "0" if per_phone_numeric_entries_used else "1"
    if quality is not None and phoneme_qualities is None:
        natural_duration_sec = float(output.shape[0]) / float(audio.sample_rate_hz)
        metadata["natural_duration_sec"] = f"{natural_duration_sec:.4f}"
    return from_numpy(output, audio, **metadata)


def synthesize_phone_segment(
    synthesizer: CoquiFragmentSynthesizer,
    phone: str,
    target_length: int | None,
    sample_rate_hz: int,
) -> np.ndarray:
    rendered = synthesizer._render_with_coqui(
        phones_to_ipa([phone]),
        sample_rate_hz,
        prephonemized=True,
    )
    if not rendered:
        raise RuntimeError(f"Coqui synthesis produced no audio for phone '{phone}'.")
    samples = np.asarray(rendered, dtype=np.float32)
    if target_length is None:
        return samples
    return resample_array(samples, target_length)


def concatenate_phone_segments(
    rendered_segments: Sequence[np.ndarray],
    sample_rate_hz: int,
) -> np.ndarray:
    if not rendered_segments:
        return np.zeros(0, dtype=np.float32)

    output = np.asarray(rendered_segments[0], dtype=np.float32).copy()
    for segment in rendered_segments[1:]:
        next_segment = np.asarray(segment, dtype=np.float32)
        if output.size == 0:
            output = next_segment.copy()
            continue
        if next_segment.size == 0:
            continue

        crossfade = min(
            output.size,
            next_segment.size,
            max(1, int(round(sample_rate_hz * 0.008))),
        )
        if crossfade <= 0:
            output = np.concatenate([output, next_segment]).astype(np.float32)
            continue

        fade_out = np.linspace(1.0, 0.0, num=crossfade, endpoint=True, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, num=crossfade, endpoint=True, dtype=np.float32)
        blended = (output[-crossfade:] * fade_out) + (next_segment[:crossfade] * fade_in)
        output = np.concatenate(
            [
                output[:-crossfade],
                blended.astype(np.float32),
                next_segment[crossfade:],
            ]
        ).astype(np.float32)
    return output


def render_blabber(
    audio: AudioBuffer,
    transcript: str,
    quality: float,
    phoneme_qualities: Sequence[float] | None = None,
    voice_mode: str = "woman",
    speaker_name: str | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
    max_phoneme_distance: float | None = None,
) -> AudioBuffer:
    ensure_coqui_backend(COQUI_ENV_NAME)
    (
        cleaned,
        source_phones,
        substituted_phones,
        preset_index,
        expansion_used,
        total_distance,
        per_phone_preset_indices,
        per_phone_distances,
        per_phone_sequences,
        per_phone_numeric_entries_used,
    ) = resolve_blabber_phone_sequence(
        transcript,
        quality,
        phoneme_qualities,
        max_phoneme_distance=max_phoneme_distance,
    )
    return render_blabber_from_resolved_phones(
        audio,
        cleaned,
        source_phones,
        substituted_phones,
        voice_mode=voice_mode,
        speaker_name=speaker_name,
        phoneme_render_backend=phoneme_render_backend,
        backend_voice=backend_voice,
        quality=quality,
        global_preset_index=preset_index,
        expansion_used=expansion_used,
        global_distance_total=total_distance,
        global_max_phoneme_distance=max_phoneme_distance if phoneme_qualities is None else None,
        per_phone_preset_indices=per_phone_preset_indices,
        per_phone_distances=per_phone_distances,
        per_phone_sequences=per_phone_sequences,
        per_phone_numeric_entries_used=per_phone_numeric_entries_used,
    )


def render_blabber_per_phoneme(
    audio: AudioBuffer,
    transcript: str,
    phoneme_qualities: Sequence[float],
    voice_mode: str = "woman",
    speaker_name: str | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
) -> AudioBuffer:
    ensure_coqui_backend(COQUI_ENV_NAME)
    (
        cleaned,
        source_phones,
        mutated_phones,
        _preset_index,
        _expansion_used,
        _total_distance,
        per_phone_preset_indices,
        per_phone_distances,
        per_phone_sequences,
        per_phone_numeric_entries_used,
    ) = resolve_blabber_phone_sequence(
        transcript,
        quality=1.0,
        phoneme_qualities=phoneme_qualities,
    )
    return render_blabber_from_resolved_phones(
        audio,
        cleaned,
        source_phones,
        mutated_phones,
        voice_mode=voice_mode,
        speaker_name=speaker_name,
        phoneme_render_backend=phoneme_render_backend,
        backend_voice=backend_voice,
        phoneme_qualities=phoneme_qualities,
        segmentation="phonetic",
        per_phone_preset_indices=per_phone_preset_indices,
        per_phone_distances=per_phone_distances,
        per_phone_sequences=per_phone_sequences,
        per_phone_numeric_entries_used=per_phone_numeric_entries_used,
    )


def render_blabber_with_segments(
    audio: AudioBuffer,
    transcript: str,
    quality: float,
    breakpoints_str: str,
    phoneme_qualities: Sequence[float] | None = None,
    voice_mode: str = "woman",
    speaker_name: str | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
    max_phoneme_distance: float | None = None,
) -> AudioBuffer:
    blabber = render_blabber(
        audio,
        transcript,
        quality,
        phoneme_qualities,
        voice_mode=voice_mode,
        speaker_name=speaker_name,
        phoneme_render_backend=phoneme_render_backend,
        backend_voice=backend_voice,
        max_phoneme_distance=max_phoneme_distance,
    )
    source_phones = blabber.metadata.get("source_phones", "").split("-") if blabber.metadata.get("source_phones") else []
    mutated_phones = (
        blabber.metadata.get("substituted_phones", "").split("-")
        if blabber.metadata.get("substituted_phones")
        else []
    )
    breakpoints, breakpoint_label = resolve_breakpoints(
        breakpoints_str,
        audio,
        transcript,
        "blabber",
        quality,
        source_phones=source_phones,
        mutated_phones=mutated_phones,
        phoneme_qualities=phoneme_qualities,
    )
    metadata = dict(blabber.metadata)
    metadata["breakpoints"] = breakpoint_label
    metadata["segmentation"] = metadata.get("segmentation", "phonetic")
    return from_numpy(to_numpy(blabber), audio, **metadata)


def render_pronunciation_safe(
    audio: AudioBuffer,
    transcript: str,
    slider_config_path: Path = DEFAULT_PRONUNCIATION_SLIDER_CONFIG,
    *,
    jibberish: float,
    clarity: float,
    timing_instability: float,
    allow_full_gibberish: bool = False,
) -> AudioBuffer:
    config = load_hybrid_config(slider_config_path)
    runtime = build_hybrid_stitching_slice(config)
    alignment = runtime.aligner.align(audio, transcript)
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(
        config,
        jibberish=jibberish,
        clarity=clarity,
        timing_instability=timing_instability,
        allow_full_gibberish=allow_full_gibberish,
    )
    plan = runtime.planner.plan(graph, controls, audio=audio)
    timing = runtime.timing_planner.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
    projected = runtime.timbre_projector.project(audio, fragments)
    stitched = runtime.assembler.assemble(audio, edited, projected)
    selected_variant = (
        plan.trace.selected_grapheme_variant
        if plan.trace is not None and plan.trace.selected_grapheme_variant
        else transcript.strip()
    )
    metadata = dict(stitched.metadata)
    metadata.update(
        {
            "augmentation": "pronunciation_safe",
            "transcript": transcript.strip(),
            "jibberish": f"{jibberish:.3f}",
            "clarity": f"{clarity:.3f}",
            "timing_instability": f"{timing_instability:.3f}",
            "allow_full_gibberish": "1" if allow_full_gibberish else "0",
            "pronunciation_similarity_score": f"{plan.similarity.score:.4f}",
            "pronunciation_similarity_threshold": f"{plan.similarity.threshold:.4f}",
            "pronunciation_fragment_count": str(len(projected)),
            "pronunciation_repair_count": str(len(plan.repairs)),
            "pronunciation_selected_word": selected_variant,
            "pronunciation_selected_word_changed": (
                "1" if selected_variant.strip().lower() != transcript.strip().lower() else "0"
            ),
            "hybrid_config": str(slider_config_path),
            "pronunciation_slider_config": str(slider_config_path),
        }
    )
    return from_numpy(to_numpy(stitched), stitched, **metadata)


def ensure_coqui_backend(env_name: str) -> None:
    completed = subprocess.run(
        resolve_python_command(env_name)
        + ["-c", "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('TTS') else 1)"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or "").strip() or (completed.stdout or "").strip()
        raise RuntimeError(
            f"`coqui-tts` is not available in conda env '{env_name}'. "
            f"Install it there before using blabber mode."
            + (f"\n\nCheck failed with:\n{details}" if details else "")
        )


def resolve_source_speaker_wav_path(audio: AudioBuffer, use_source_speaker_wav: bool) -> str | None:
    if not use_source_speaker_wav:
        return None
    source_path = audio.metadata.get("source_path")
    if not source_path:
        raise ValueError("Source speaker_wav was requested, but the loaded audio does not expose a source path.")
    path = Path(source_path)
    if not path.is_file():
        raise ValueError(f"Source speaker_wav was requested, but '{source_path}' does not exist.")
    return str(path)


class SpeechDistortionGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Speech Distortion GUI")
        self.root.geometry("760x1080")
        self.scroll_canvas: tk.Canvas | None = None
        self.scroll_container: ttk.Frame | None = None

        self.reader = WavAudioReader()
        self.writer = WavAudioWriter()
        self.current_audio: AudioBuffer | None = None
        self.rendered_audio: AudioBuffer | None = None
        self.preview_path: Path | None = None
        self.phoneme_quality_vars: list[tk.DoubleVar] = []
        self.detected_phones: list[str] = []
        self.cached_blabber_sequence: dict[str, object] | None = None
        self.live_blabber_sequence_after_id: str | None = None
        self.live_pronunciation_preview_after_id: str | None = None
        self.hybrid_planning_runtime = None

        self.audio_path_var = tk.StringVar(value=str(DEFAULT_INPUT if DEFAULT_INPUT.exists() else ""))
        self.mode_var = tk.StringVar(value="static_noise")
        self.phone_style_noise_var = tk.StringVar(value="white")
        self.quality_mode_var = tk.StringVar(value="global")
        self.quality_var = tk.DoubleVar(value=0.75)
        self.jibberish_var = tk.DoubleVar(value=0.4)
        self.clarity_var = tk.DoubleVar(value=0.4)
        self.timing_instability_var = tk.DoubleVar(value=0.35)
        self.allow_full_gibberish_var = tk.BooleanVar(value=False)
        self.max_phoneme_distance_var = tk.StringVar(value=DEFAULT_GLOBAL_BLABBER_MAX_PHONEME_DISTANCE)
        self.transcript_var = tk.StringVar(value="bath")
        self.breakpoints_var = tk.StringVar(value="auto")
        self.voice_mode_var = tk.StringVar(value="woman")
        self.male_speaker_var = tk.StringVar(value=DEFAULT_MALE_COQUI_SPEAKER)
        self.phoneme_render_backend_var = tk.StringVar(value=DEFAULT_PHONEME_RENDER_BACKEND)
        self.kokoro_voice_var = tk.StringVar(value=DEFAULT_KOKORO_VOICE)
        self.style_transfer_backend_var = tk.StringVar(value=DEFAULT_STYLE_TRANSFER_BACKEND)
        self.style_transfer_target_voice_var = tk.StringVar(value="")
        self.style_transfer_presets_path = DEFAULT_STYLE_TRANSFER_PRESETS_PATH
        self.style_transfer_preset_names: list[str] = []
        self.status_var = tk.StringVar(value="Choose a WAV file and render an augmentation.")
        self.subtitle_var = tk.StringVar(value=subtitle_text_for_mode(self.mode_var.get()))
        self.slider_label_var = tk.StringVar(value=slider_caption_for_mode(self.mode_var.get()))

        self._build_ui()
        self.transcript_var.trace_add("write", self._on_transcript_change)
        self.quality_var.trace_add("write", self._on_blabber_inputs_changed)
        self.jibberish_var.trace_add("write", self._on_blabber_inputs_changed)
        self.clarity_var.trace_add("write", self._on_blabber_inputs_changed)
        self.timing_instability_var.trace_add("write", self._on_blabber_inputs_changed)
        self.allow_full_gibberish_var.trace_add("write", self._on_blabber_inputs_changed)
        self.mode_var.trace_add("write", self._on_mode_changed)
        self.quality_mode_var.trace_add("write", self._on_blabber_inputs_changed)
        self.max_phoneme_distance_var.trace_add("write", self._on_blabber_inputs_changed)
        self.voice_mode_var.trace_add("write", self._on_voice_mode_changed)
        self.male_speaker_var.trace_add("write", self._on_blabber_inputs_changed)
        self.phoneme_render_backend_var.trace_add("write", self._on_blabber_inputs_changed)
        self.kokoro_voice_var.trace_add("write", self._on_blabber_inputs_changed)
        self.style_transfer_backend_var.trace_add("write", self._on_style_transfer_backend_changed)
        self.style_transfer_target_voice_var.trace_add("write", self._on_blabber_inputs_changed)
        self._refresh_style_transfer_presets()
        if self.audio_path_var.get():
            self._load_audio(Path(self.audio_path_var.get()))
        self._refresh_phoneme_controls()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        container = ttk.Frame(self.scroll_canvas, padding=16)
        self.scroll_container = container
        self.scroll_canvas_window = self.scroll_canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind("<Configure>", self._on_scrollable_frame_configure)
        self.scroll_canvas.bind("<Configure>", self._on_scrollable_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

        title = ttk.Label(
            container,
            text="Speech Distortion GUI",
            font=("Segoe UI", 15, "bold"),
        )
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            container,
            textvariable=self.subtitle_var,
        )
        subtitle.pack(anchor=tk.W, pady=(4, 16))

        file_row = ttk.Frame(container)
        file_row.pack(fill=tk.X)
        ttk.Label(file_row, text="Input WAV", width=14).pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.audio_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_row, text="Browse", command=self._browse_audio).pack(side=tk.LEFT, padx=(8, 0))

        transcript_row = ttk.Frame(container)
        transcript_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(transcript_row, text="Transcript", width=14).pack(side=tk.LEFT)
        self.transcript_entry = ttk.Entry(transcript_row, textvariable=self.transcript_var)
        self.transcript_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        breakpoint_row = ttk.Frame(container)
        breakpoint_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(breakpoint_row, text="Breakpoints", width=14).pack(side=tk.LEFT)
        self.breakpoint_entry = ttk.Entry(breakpoint_row, textvariable=self.breakpoints_var)
        self.breakpoint_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mode_frame = ttk.LabelFrame(container, text="Augmentation", padding=12)
        mode_frame.pack(fill=tk.X, pady=(16, 0))
        for text, value in (
            ("Static noise", "static_noise"),
            ("Volume dropout", "volume_dropout"),
            ("Phone style", "phone_style"),
            ("Blabber", "blabber"),
            ("Pronunciation safe", "pronunciation_safe"),
        ):
            ttk.Radiobutton(
                mode_frame,
                text=text,
                value=value,
                variable=self.mode_var,
                command=self._update_mode_state,
            ).pack(anchor=tk.W)

        phone_style_row = ttk.Frame(container)
        phone_style_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(phone_style_row, text="Style noise", width=14).pack(side=tk.LEFT)
        self.phone_style_combo = ttk.Combobox(
            phone_style_row,
            textvariable=self.phone_style_noise_var,
            values=("white", "pink"),
            state="readonly",
        )
        self.phone_style_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        voice_row = ttk.Frame(container)
        voice_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(voice_row, text="Voice", width=14).pack(side=tk.LEFT)
        self.voice_mode_combo = ttk.Combobox(
            voice_row,
            textvariable=self.voice_mode_var,
            values=BLABBER_VOICE_MODES,
            state="readonly",
        )
        self.voice_mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        male_speaker_row = ttk.Frame(container)
        male_speaker_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(male_speaker_row, text="Male speaker", width=14).pack(side=tk.LEFT)
        self.male_speaker_combo = ttk.Combobox(
            male_speaker_row,
            textvariable=self.male_speaker_var,
            values=MALE_COQUI_SPEAKER_OPTIONS,
            state="readonly",
        )
        self.male_speaker_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        phoneme_render_backend_row = ttk.Frame(container)
        phoneme_render_backend_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(phoneme_render_backend_row, text="Phoneme TTS", width=14).pack(side=tk.LEFT)
        self.phoneme_render_backend_combo = ttk.Combobox(
            phoneme_render_backend_row,
            textvariable=self.phoneme_render_backend_var,
            values=PHONEME_RENDER_BACKENDS,
            state="readonly",
        )
        self.phoneme_render_backend_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        kokoro_voice_row = ttk.Frame(container)
        kokoro_voice_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(kokoro_voice_row, text="Kokoro voice", width=14).pack(side=tk.LEFT)
        self.kokoro_voice_combo = ttk.Combobox(
            kokoro_voice_row,
            textvariable=self.kokoro_voice_var,
            values=KOKORO_VOICE_OPTIONS,
            state="readonly",
        )
        self.kokoro_voice_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        style_transfer_backend_row = ttk.Frame(container)
        style_transfer_backend_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(style_transfer_backend_row, text="Style transfer", width=14).pack(side=tk.LEFT)
        self.style_transfer_backend_combo = ttk.Combobox(
            style_transfer_backend_row,
            textvariable=self.style_transfer_backend_var,
            values=STYLE_TRANSFER_BACKENDS,
            state="readonly",
        )
        self.style_transfer_backend_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        style_transfer_target_row = ttk.Frame(container)
        style_transfer_target_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(style_transfer_target_row, text="Target preset", width=14).pack(side=tk.LEFT)
        self.style_transfer_target_combo = ttk.Combobox(
            style_transfer_target_row,
            textvariable=self.style_transfer_target_voice_var,
            values=(),
            state="readonly",
        )
        self.style_transfer_target_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        phoneme_frame = ttk.LabelFrame(container, text="Phoneme Quality", padding=12)
        phoneme_frame.pack(fill=tk.X, pady=(12, 0))
        quality_mode_row = ttk.Frame(phoneme_frame)
        quality_mode_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(quality_mode_row, text="Quality mode", width=14).pack(side=tk.LEFT)
        ttk.Radiobutton(
            quality_mode_row,
            text="Global",
            value="global",
            variable=self.quality_mode_var,
            command=self._update_quality_mode_state,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            quality_mode_row,
            text="Per-phoneme",
            value="per_phoneme",
            variable=self.quality_mode_var,
            command=self._update_quality_mode_state,
        ).pack(side=tk.LEFT, padx=(12, 0))

        max_distance_row = ttk.Frame(phoneme_frame)
        max_distance_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(max_distance_row, text="Max global dist", width=14).pack(side=tk.LEFT)
        self.max_phoneme_distance_combo = ttk.Combobox(
            max_distance_row,
            textvariable=self.max_phoneme_distance_var,
            values=GLOBAL_BLABBER_MAX_PHONEME_DISTANCE_VALUES,
            width=8,
            state="readonly",
        )
        self.max_phoneme_distance_combo.pack(side=tk.LEFT)
        self.phoneme_controls_frame = ttk.Frame(phoneme_frame)
        self.phoneme_controls_frame.pack(fill=tk.X, expand=True)

        slider_frame = ttk.Frame(container)
        slider_frame.pack(fill=tk.X, pady=(16, 0))
        ttk.Label(slider_frame, textvariable=self.slider_label_var, width=14).pack(side=tk.LEFT)
        self.global_quality_scale = ttk.Scale(
            slider_frame,
            from_=0.0,
            to=1.0,
            variable=self.quality_var,
            orient=tk.HORIZONTAL,
            command=self._sync_quality_label,
        )
        self.global_quality_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.quality_label = ttk.Label(slider_frame, width=6, text="0.75")
        self.quality_label.pack(side=tk.LEFT, padx=(8, 0))

        self.pronunciation_slider_frame = ttk.LabelFrame(container, text="Pronunciation Sliders", padding=12)
        self.pronunciation_slider_frame.pack(fill=tk.X, pady=(12, 0))
        self._build_pronunciation_slider_row(
            self.pronunciation_slider_frame,
            "Jibberish",
            self.jibberish_var,
        )
        self._build_pronunciation_slider_row(
            self.pronunciation_slider_frame,
            "Clarity",
            self.clarity_var,
        )
        self._build_pronunciation_slider_row(
            self.pronunciation_slider_frame,
            "Timing instability",
            self.timing_instability_var,
        )
        ttk.Checkbutton(
            self.pronunciation_slider_frame,
            text="Allow full gibberish override",
            variable=self.allow_full_gibberish_var,
        ).pack(anchor=tk.W, pady=(8, 0))

        info_frame = ttk.LabelFrame(container, text="Status", padding=12)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        self.output_text = ScrolledText(info_frame, wrap=tk.WORD, height=12)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.insert("1.0", self.status_var.get())

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(button_row, text="Render", command=self._render).pack(side=tk.LEFT)
        self.generate_sequence_button = ttk.Button(
            button_row,
            text="Generate Sequence",
            command=self._generate_blabber_sequence,
        )
        self.generate_sequence_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Preview", command=self._preview).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Save As", command=self._save_as).pack(side=tk.LEFT, padx=(8, 0))

        self._sync_quality_label()
        self._update_mode_state()
        self._update_quality_mode_state()

    def _build_pronunciation_slider_row(
        self,
        parent: ttk.Frame,
        label_text: str,
        variable: tk.DoubleVar,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label_text, width=18).pack(side=tk.LEFT)
        scale = ttk.Scale(
            row,
            from_=0.0,
            to=1.0,
            variable=variable,
            orient=tk.HORIZONTAL,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        value_label = ttk.Label(row, width=6, text=f"{variable.get():.2f}")
        value_label.pack(side=tk.LEFT, padx=(8, 0))
        scale.configure(
            command=lambda _value, var=variable, label=value_label: label.configure(text=f"{var.get():.2f}")
        )

    def _on_scrollable_frame_configure(self, _event: tk.Event[tk.Misc]) -> None:
        if self.scroll_canvas is None or self.scroll_container is None:
            return
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_scrollable_canvas_configure(self, event: tk.Event[tk.Misc]) -> None:
        if self.scroll_canvas is None:
            return
        self.scroll_canvas.itemconfigure(self.scroll_canvas_window, width=event.width)

    def _should_scroll_main_canvas(self) -> bool:
        if self.scroll_canvas is None:
            return False
        focus_widget = self.root.focus_get()
        return not isinstance(focus_widget, tk.Text)

    def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> None:
        if self.scroll_canvas is None or not self._should_scroll_main_canvas():
            return
        if getattr(event, "delta", 0):
            units = -int(event.delta / 120)
        elif getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            return
        self.scroll_canvas.yview_scroll(units, "units")

    def _sync_quality_label(self, *_args: object) -> None:
        self.quality_label.configure(
            text=slider_value_text_for_mode(self.mode_var.get(), self.quality_var.get())
        )

    def _on_transcript_change(self, *_args: object) -> None:
        self._invalidate_blabber_sequence()
        self._refresh_phoneme_controls()
        self._schedule_live_blabber_sequence_update()
        self._schedule_live_pronunciation_preview()

    def _on_mode_changed(self, *_args: object) -> None:
        self._invalidate_blabber_sequence()
        self._schedule_live_blabber_sequence_update()
        self._schedule_live_pronunciation_preview()

    def _on_blabber_inputs_changed(self, *_args: object) -> None:
        self._invalidate_blabber_sequence()
        self._schedule_live_blabber_sequence_update()
        self._schedule_live_pronunciation_preview()

    def _on_voice_mode_changed(self, *_args: object) -> None:
        if self.voice_mode_var.get().strip().lower() != "woman":
            self.style_transfer_backend_var.set(STYLE_TRANSFER_BACKEND_NONE)
        self._update_mode_state()
        self._on_blabber_inputs_changed()

    def _on_style_transfer_backend_changed(self, *_args: object) -> None:
        self._update_mode_state()
        self._on_blabber_inputs_changed()

    def _refresh_style_transfer_presets(self) -> None:
        presets = male_style_transfer_presets(self.style_transfer_presets_path)
        self.style_transfer_preset_names = [preset.name for preset in presets]
        self.style_transfer_target_combo.configure(values=self.style_transfer_preset_names)
        current_value = self.style_transfer_target_voice_var.get().strip()
        if current_value not in self.style_transfer_preset_names:
            self.style_transfer_target_voice_var.set(self.style_transfer_preset_names[0] if self.style_transfer_preset_names else "")

    def _refresh_phoneme_controls(self) -> None:
        for child in self.phoneme_controls_frame.winfo_children():
            child.destroy()
        transcript = self.transcript_var.get().strip()
        if not transcript:
            self.detected_phones = []
            self.phoneme_quality_vars = []
            ttk.Label(
                self.phoneme_controls_frame,
                text="Enter a transcript to detect phoneme segments.",
            ).pack(anchor=tk.W)
            return

        g2p = HeuristicEnglishG2P()
        self.detected_phones = g2p.phonemize_word(transcript)
        self.phoneme_quality_vars = []
        if not self.detected_phones:
            ttk.Label(
                self.phoneme_controls_frame,
                text="No phonemes detected for the current transcript.",
            ).pack(anchor=tk.W)
            return

        for index, phone in enumerate(self.detected_phones):
            row = ttk.Frame(self.phoneme_controls_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{index + 1}. {phone}", width=10).pack(side=tk.LEFT)
            quality_var = tk.DoubleVar(value=0.5)
            self.phoneme_quality_vars.append(quality_var)
            scale = ttk.Scale(
                row,
                from_=0.0,
                to=1.0,
                variable=quality_var,
                orient=tk.HORIZONTAL,
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            value_label = ttk.Label(row, width=6, text="0.50")
            value_label.pack(side=tk.LEFT, padx=(8, 0))
            quality_var.trace_add("write", self._on_blabber_inputs_changed)
            scale.configure(
                command=lambda _value, var=quality_var, label=value_label: label.configure(text=f"{var.get():.2f}")
            )
        self._update_quality_mode_state()

    def _update_quality_mode_state(self) -> None:
        use_per_phoneme = self.quality_mode_var.get() == "per_phoneme"
        pronunciation_mode = self.mode_var.get() == "pronunciation_safe"
        if pronunciation_mode:
            use_per_phoneme = False
        if use_per_phoneme:
            self.global_quality_scale.state(["disabled"])
            self.max_phoneme_distance_combo.state(["disabled"])
        else:
            if pronunciation_mode:
                self.global_quality_scale.state(["disabled"])
            else:
                self.global_quality_scale.state(["!disabled"])
            if self.mode_var.get() == "blabber":
                self.max_phoneme_distance_combo.state(["!disabled", "readonly"])
            else:
                self.max_phoneme_distance_combo.state(["disabled"])
        for child in self.phoneme_controls_frame.winfo_children():
            for grandchild in child.winfo_children():
                if isinstance(grandchild, ttk.Scale):
                    if use_per_phoneme and not pronunciation_mode:
                        grandchild.state(["!disabled"])
                    else:
                        grandchild.state(["disabled"])

    def _update_mode_state(self) -> None:
        mode = self.mode_var.get()
        self.slider_label_var.set(slider_caption_for_mode(mode))
        self.subtitle_var.set(subtitle_text_for_mode(mode))
        self._sync_quality_label()
        self.transcript_entry.state(["!disabled"])
        pronunciation_mode = mode == "pronunciation_safe"
        if mode == "phone_style":
            self.phone_style_combo.state(["!disabled", "readonly"])
        else:
            self.phone_style_combo.state(["disabled"])
        if mode == "blabber":
            self.voice_mode_combo.state(["!disabled", "readonly"])
            self.generate_sequence_button.state(["!disabled"])
            self.style_transfer_backend_combo.state(["!disabled", "readonly"])
            self.phoneme_render_backend_combo.state(["!disabled", "readonly"])
        else:
            self.voice_mode_combo.state(["disabled"])
            self.generate_sequence_button.state(["disabled"])
            self.style_transfer_backend_combo.state(["disabled"])
            self.phoneme_render_backend_combo.state(["disabled"])
        if mode == "blabber" and self.voice_mode_var.get() == "man":
            self.male_speaker_combo.state(["!disabled", "readonly"])
        else:
            self.male_speaker_combo.state(["disabled"])
        if (
            mode == "blabber"
            and self.voice_mode_var.get() == "woman"
            and self.phoneme_render_backend_var.get().strip() == PHONEME_RENDER_BACKEND_KOKORO
        ):
            self.kokoro_voice_combo.state(["!disabled", "readonly"])
        else:
            self.kokoro_voice_combo.state(["disabled"])
        if (
            mode == "blabber"
            and self.style_transfer_backend_var.get() != STYLE_TRANSFER_BACKEND_NONE
            and self.voice_mode_var.get() == "woman"
            and self.style_transfer_preset_names
        ):
            self.style_transfer_target_combo.state(["!disabled", "readonly"])
        else:
            self.style_transfer_target_combo.state(["disabled"])
        if pronunciation_mode:
            self.pronunciation_slider_frame.state(["!disabled"])
        else:
            self.pronunciation_slider_frame.state(["disabled"])
        self._update_quality_mode_state()

    def _invalidate_blabber_sequence(self) -> None:
        self.cached_blabber_sequence = None

    def _cancel_live_blabber_sequence_update(self) -> None:
        if self.live_blabber_sequence_after_id is None:
            return
        try:
            self.root.after_cancel(self.live_blabber_sequence_after_id)
        except tk.TclError:
            pass
        self.live_blabber_sequence_after_id = None

    def _schedule_live_blabber_sequence_update(self) -> None:
        self._cancel_live_blabber_sequence_update()
        if self.mode_var.get() != "blabber":
            return
        self.live_blabber_sequence_after_id = self.root.after(120, self._refresh_live_blabber_sequence)

    def _cancel_live_pronunciation_preview(self) -> None:
        if self.live_pronunciation_preview_after_id is None:
            return
        try:
            self.root.after_cancel(self.live_pronunciation_preview_after_id)
        except tk.TclError:
            pass
        self.live_pronunciation_preview_after_id = None

    def _schedule_live_pronunciation_preview(self) -> None:
        self._cancel_live_pronunciation_preview()
        if self.mode_var.get() != "pronunciation_safe":
            return
        self.live_pronunciation_preview_after_id = self.root.after(120, self._refresh_live_pronunciation_preview)

    def _refresh_live_pronunciation_preview(self) -> None:
        self.live_pronunciation_preview_after_id = None
        if self.mode_var.get() != "pronunciation_safe":
            return
        try:
            preview = self._compute_pronunciation_preview()
        except Exception as exc:
            self._set_output(f"Live pronunciation preview unavailable\n\n{exc}")
            return
        self._set_output(self._format_pronunciation_preview_status(preview))

    def _refresh_live_blabber_sequence(self) -> None:
        self.live_blabber_sequence_after_id = None
        if self.mode_var.get() != "blabber":
            return
        try:
            sequence_payload = self._compute_blabber_sequence()
        except Exception as exc:
            self.cached_blabber_sequence = None
            self._set_output(f"Live blabber sequence unavailable\n\n{exc}")
            return
        self.cached_blabber_sequence = sequence_payload
        self._set_output(self._format_blabber_sequence_status(sequence_payload, cached=True, live=True))

    def _current_global_max_phoneme_distance(self) -> float:
        try:
            return max(0.0, float(self.max_phoneme_distance_var.get()))
        except (TypeError, ValueError):
            self.max_phoneme_distance_var.set(DEFAULT_GLOBAL_BLABBER_MAX_PHONEME_DISTANCE)
            return float(DEFAULT_GLOBAL_BLABBER_MAX_PHONEME_DISTANCE)

    def _current_blabber_controls(self) -> tuple[str, float, str, tuple[float, ...], float | None]:
        use_per_phoneme = self.quality_mode_var.get() == "per_phoneme"
        quality = 1.0 if use_per_phoneme else max(0.0, min(1.0, float(self.quality_var.get())))
        phoneme_qualities = (
            tuple(float(var.get()) for var in self.phoneme_quality_vars)
            if use_per_phoneme
            else tuple()
        )
        max_phoneme_distance = None if use_per_phoneme else self._current_global_max_phoneme_distance()
        return (
            self.transcript_var.get().strip(),
            quality,
            self.quality_mode_var.get(),
            phoneme_qualities,
            max_phoneme_distance,
        )

    def _current_male_speaker(self) -> str | None:
        voice_mode = normalize_blabber_voice_mode(self.voice_mode_var.get())
        if voice_mode != "man":
            return None
        speaker_name = self.male_speaker_var.get().strip() or DEFAULT_MALE_COQUI_SPEAKER
        if speaker_name not in MALE_COQUI_SPEAKER_OPTIONS:
            self.male_speaker_var.set(DEFAULT_MALE_COQUI_SPEAKER)
            return DEFAULT_MALE_COQUI_SPEAKER
        return speaker_name

    def _current_style_transfer_controls(self) -> tuple[str, str]:
        backend = self.style_transfer_backend_var.get().strip() or STYLE_TRANSFER_BACKEND_NONE
        if self.voice_mode_var.get().strip().lower() != "woman":
            return STYLE_TRANSFER_BACKEND_NONE, ""
        if backend == STYLE_TRANSFER_BACKEND_NONE:
            return backend, ""
        target_voice = self.style_transfer_target_voice_var.get().strip()
        if not target_voice:
            raise ValueError(
                "No male style transfer preset is configured. "
                f"Add presets to '{self.style_transfer_presets_path}'."
            )
        if target_voice not in self.style_transfer_preset_names:
            raise ValueError(
                f"Unknown style transfer preset '{target_voice}'. "
                f"Update '{self.style_transfer_presets_path}' or choose another preset."
            )
        return backend, target_voice

    def _current_backend_voice(self) -> str | None:
        if self.voice_mode_var.get().strip().lower() != "woman":
            return None
        backend = self.phoneme_render_backend_var.get().strip()
        if backend != PHONEME_RENDER_BACKEND_KOKORO:
            return None
        voice_name = self.kokoro_voice_var.get().strip().lower() or DEFAULT_KOKORO_VOICE
        if voice_name not in KOKORO_VOICE_OPTIONS:
            self.kokoro_voice_var.set(DEFAULT_KOKORO_VOICE)
            return DEFAULT_KOKORO_VOICE
        return voice_name

    def _get_hybrid_planning_runtime(self):
        if self.hybrid_planning_runtime is None:
            config = load_hybrid_config(DEFAULT_PRONUNCIATION_SLIDER_CONFIG)
            self.hybrid_planning_runtime = build_hybrid_planning_slice(config)
        return self.hybrid_planning_runtime

    def _compute_pronunciation_preview(self) -> dict[str, object]:
        transcript = self.transcript_var.get().strip()
        if not transcript:
            raise ValueError("Enter a transcript to preview pronunciation-safe candidates.")
        path_text = self.audio_path_var.get().strip()
        if not path_text:
            raise ValueError("Choose a WAV file to preview pronunciation-safe candidates.")
        path = Path(path_text)
        if self.current_audio is None or self.current_audio.metadata.get("source_path") != str(path):
            self._load_audio(path)
        if self.current_audio is None:
            raise ValueError("Input audio is not available.")

        runtime = self._get_hybrid_planning_runtime()
        controls = build_slider_controls(
            runtime.config,
            jibberish=max(0.0, min(1.0, float(self.jibberish_var.get()))),
            clarity=max(0.0, min(1.0, float(self.clarity_var.get()))),
            timing_instability=max(0.0, min(1.0, float(self.timing_instability_var.get()))),
            allow_full_gibberish=bool(self.allow_full_gibberish_var.get()),
        )
        alignment = runtime.aligner.align(self.current_audio, transcript)
        graph = runtime.feature_extractor.build_phone_graph(alignment)
        plan = runtime.planner.plan(graph, controls, audio=self.current_audio)
        accepted_candidates = [candidate for candidate in plan.candidates if candidate.accepted]
        accepted_candidates.sort(
            key=lambda item: (
                item.word_index,
                item.grapheme == item.word,
                -item.pronunciation_similarity,
                item.phoneme_distance,
                item.grapheme,
            )
        )
        return {
            "transcript": transcript,
            "selected_word": plan.trace.selected_grapheme_variant if plan.trace is not None else transcript,
            "selected_phones": list(plan.trace.selected_phones) if plan.trace is not None else [],
            "similarity_score": plan.similarity.score,
            "similarity_threshold": plan.similarity.threshold,
            "accepted_candidates": accepted_candidates[:10],
            "rejected_count": len([candidate for candidate in plan.candidates if not candidate.accepted]),
            "jibberish": controls.jibberish,
            "clarity": controls.clarity,
            "timing_instability": controls.timing_instability,
        }

    def _format_pronunciation_preview_status(self, preview: dict[str, object]) -> str:
        selected_word = str(preview["selected_word"])
        transcript = str(preview["transcript"])
        lines = [
            "Live pronunciation-safe preview",
            f"Transcript: {transcript}",
            f"Selected word: {selected_word}",
            f"Jibberish: {float(preview['jibberish']):.2f}",
            f"Clarity: {float(preview['clarity']):.2f}",
            f"Timing instability: {float(preview['timing_instability']):.2f}",
            (
                "Similarity: "
                f"{float(preview['similarity_score']):.4f} / {float(preview['similarity_threshold']):.4f}"
            ),
        ]
        selected_phones = list(preview.get("selected_phones", []))
        if selected_phones:
            lines.append(f"Selected phones: {'-'.join(str(phone) for phone in selected_phones)}")
        accepted_candidates = list(preview.get("accepted_candidates", []))
        if accepted_candidates:
            lines.append("Accepted candidates:")
            for candidate in accepted_candidates[:6]:
                lines.append(
                    "  {0}  sim={1:.3f} phone_dist={2:.3f} graph_dist={3:.3f}".format(
                        candidate.grapheme,
                        candidate.pronunciation_similarity,
                        candidate.phoneme_distance,
                        candidate.grapheme_distance,
                    )
                )
        lines.append(f"Rejected candidates: {int(preview.get('rejected_count', 0))}")
        return "\n".join(lines)

    def _compute_blabber_sequence(self) -> dict[str, object]:
        transcript, quality, quality_mode, phoneme_qualities, \
                max_phoneme_distance = self._current_blabber_controls()
        use_per_phoneme = quality_mode == "per_phoneme"
        resolved_phoneme_qualities = list(phoneme_qualities) if use_per_phoneme else None
        (
            cleaned,
            source_phones,
            mutated_phones,
            preset_index,
            expansion_used,
            total_distance,
            per_phone_preset_indices,
            per_phone_distances,
            per_phone_sequences,
            per_phone_numeric_entries_used,
        ) = resolve_blabber_phone_sequence(
            transcript,
            quality,
            resolved_phoneme_qualities,
            max_phoneme_distance=max_phoneme_distance,
        )
        return {
            "transcript": cleaned,
            "quality": quality,
            "quality_mode": quality_mode,
            "phoneme_qualities": phoneme_qualities,
            "source_phones": source_phones,
            "mutated_phones": mutated_phones,
            "global_preset_index": preset_index,
            "expansion_used": expansion_used,
            "global_distance_total": total_distance,
            "whole_word_distance_total": total_distance,
            "global_max_phoneme_distance": max_phoneme_distance,
            "per_phone_preset_indices": per_phone_preset_indices,
            "per_phone_distances": per_phone_distances,
            "per_phone_sequences": per_phone_sequences,
            "per_phone_numeric_entries_used": per_phone_numeric_entries_used,
        }

    def _get_cached_blabber_sequence(self) -> dict[str, object] | None:
        if self.cached_blabber_sequence is None:
            return None
        transcript, quality, quality_mode, phoneme_qualities, max_phoneme_distance = self._current_blabber_controls()
        if (
            self.cached_blabber_sequence.get("transcript") != transcript.strip()
            or self.cached_blabber_sequence.get("quality") != quality
            or self.cached_blabber_sequence.get("quality_mode") != quality_mode
            or self.cached_blabber_sequence.get("phoneme_qualities") != phoneme_qualities
            or self.cached_blabber_sequence.get("global_max_phoneme_distance") != max_phoneme_distance
        ):
            self.cached_blabber_sequence = None
            return None
        return self.cached_blabber_sequence

    def _format_blabber_sequence_status(
        self,
        sequence_payload: dict[str, object],
        cached: bool,
        live: bool = False,
    ) -> str:
        quality_mode = str(sequence_payload["quality_mode"])
        lines = [
            "Live blabber sequence" if live else "Generated blabber sequence",
            f"Transcript: {sequence_payload['transcript']}",
            f"Quality mode: {'per-phoneme' if quality_mode == 'per_phoneme' else 'global'}",
            f"Source phones: {'-'.join(sequence_payload['source_phones'])}",
            f"Mutated phones: {'-'.join(sequence_payload['mutated_phones'])}",
            f"Cached for render: {'yes' if cached else 'no'}",
        ]
        if quality_mode == "per_phoneme":
            lines.append(
                "Phoneme qualities: "
                + ",".join(f"{clamp_unit(float(value)):.2f}" for value in sequence_payload["phoneme_qualities"])
            )
            if sequence_payload.get("per_phone_preset_indices") is not None:
                lines.append(
                    "Per-phone presets: "
                    + ",".join(str(int(value)) for value in sequence_payload["per_phone_preset_indices"])
                )
            if sequence_payload.get("per_phone_distances") is not None:
                lines.append(
                    "Per-phone distances: "
                    + ",".join(f"{float(value):.4f}" for value in sequence_payload["per_phone_distances"])
                )
            if sequence_payload.get("whole_word_distance_total") is not None:
                lines.append(f"Whole-word distance: {float(sequence_payload['whole_word_distance_total']):.4f}")
            if sequence_payload.get("per_phone_sequences") is not None:
                lines.append(
                    "Per-phone sequences: "
                    + ";".join("-".join(sequence) for sequence in sequence_payload["per_phone_sequences"])
                )
            if sequence_payload.get("per_phone_numeric_entries_used") is not None:
                lines.append(
                    "Reference entries: "
                    + (
                        "numeric candidate distances"
                        if bool(sequence_payload["per_phone_numeric_entries_used"])
                        else "legacy fallback ordering"
                    )
                )
                if not bool(sequence_payload["per_phone_numeric_entries_used"]):
                    lines.append("Reference warning: regenerate phoneme_distances.json for smoother per-phoneme ladders")
        else:
            lines.append(f"Quality: {float(sequence_payload['quality']):.2f}")
            if sequence_payload.get("global_max_phoneme_distance") is not None:
                lines.append(f"Max global distance: {float(sequence_payload['global_max_phoneme_distance']):.1f}")
            if sequence_payload["global_preset_index"] is not None:
                lines.append(f"Global preset: {int(sequence_payload['global_preset_index'])}")
            if sequence_payload.get("global_distance_total") is not None:
                lines.append(f"Total distance: {float(sequence_payload['global_distance_total']):.4f}")
            lines.append(f"Expansion used: {'yes' if bool(sequence_payload['expansion_used']) else 'no'}")
        return "\n".join(lines)

    def _generate_blabber_sequence(self) -> None:
        if self.mode_var.get() != "blabber":
            messagebox.showerror("Wrong mode", "Generate Sequence is only available in Blabber mode.")
            return
        self._cancel_live_blabber_sequence_update()
        try:
            sequence_payload = self._compute_blabber_sequence()
        except Exception as exc:
            tb = traceback.format_exc()
            self._set_output(f"Sequence generation failed\n\n{exc}\n\nTraceback:\n{tb}")
            messagebox.showerror("Sequence generation failed", str(exc))
            return
        self.cached_blabber_sequence = sequence_payload
        self._set_output(self._format_blabber_sequence_status(sequence_payload, cached=True))

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        self.audio_path_var.set(path)
        self._load_audio(Path(path))

    def _load_audio(self, path: Path) -> None:
        try:
            self.current_audio = self.reader.read(str(path))
            duration = len(self.current_audio.samples) / float(self.current_audio.sample_rate_hz)
            self.hybrid_planning_runtime = None
            self._set_output(
                f"Loaded {path.name}\n"
                f"Sample rate: {self.current_audio.sample_rate_hz} Hz\n"
                f"Duration: {duration:.2f} s"
            )
        except Exception as exc:
            self.current_audio = None
            self._set_output(f"Failed to load {path}: {exc}")

    def _render(self) -> None:
        self._cancel_live_blabber_sequence_update()
        self._cancel_live_pronunciation_preview()
        path_text = self.audio_path_var.get().strip()
        if not path_text:
            messagebox.showerror("Missing file", "Choose a WAV file first.")
            return

        path = Path(path_text)
        if self.current_audio is None or self.current_audio.metadata.get("source_path") != str(path):
            self._load_audio(path)
        if self.current_audio is None:
            return

        use_per_phoneme = self.quality_mode_var.get() == "per_phoneme"
        quality = 1.0 if use_per_phoneme else max(0.0, min(1.0, float(self.quality_var.get())))
        effect_quality = 0.0 if use_per_phoneme else quality
        mode = self.mode_var.get()
        breakpoints = self.breakpoints_var.get()
        transcript = self.transcript_var.get()
        phoneme_qualities = [float(var.get()) for var in self.phoneme_quality_vars] if use_per_phoneme else None
        voice_mode = normalize_blabber_voice_mode(self.voice_mode_var.get())
        male_speaker = self._current_male_speaker()
        phoneme_render_backend = self.phoneme_render_backend_var.get().strip() or DEFAULT_PHONEME_RENDER_BACKEND
        backend_voice = self._current_backend_voice()
        style_transfer_backend, style_transfer_target_voice = self._current_style_transfer_controls()
        try:
            if mode == "static_noise":
                if use_per_phoneme:
                    self.rendered_audio = augment_by_phoneme_segments(
                        self.current_audio,
                        transcript,
                        phoneme_qualities or [],
                        "static_noise",
                    )
                else:
                    effected = apply_static_noise(self.current_audio, effect_quality)
                    resolved_breakpoints, breakpoint_label = resolve_breakpoints(
                        breakpoints,
                        self.current_audio,
                        transcript,
                        "static_noise",
                        quality,
                        phoneme_qualities=phoneme_qualities,
                    )
                    output = apply_time_envelope_breakpoints(
                        to_numpy(self.current_audio),
                        to_numpy(effected),
                        resolved_breakpoints,
                    )
                    self.rendered_audio = from_numpy(
                        output,
                        self.current_audio,
                        augmentation="static_noise",
                        quality=f"{quality:.3f}",
                        breakpoints=breakpoint_label,
                    )
            elif mode == "volume_dropout":
                if use_per_phoneme:
                    self.rendered_audio = augment_by_phoneme_segments(
                        self.current_audio,
                        transcript,
                        phoneme_qualities or [],
                        "volume_dropout",
                    )
                else:
                    effected = apply_volume_dropout(self.current_audio, effect_quality)
                    resolved_breakpoints, breakpoint_label = resolve_breakpoints(
                        breakpoints,
                        self.current_audio,
                        transcript,
                        "volume_dropout",
                        quality,
                        phoneme_qualities=phoneme_qualities,
                    )
                    output = apply_time_envelope_breakpoints(
                        to_numpy(self.current_audio),
                        to_numpy(effected),
                        resolved_breakpoints,
                    )
                    self.rendered_audio = from_numpy(
                        output,
                        self.current_audio,
                        augmentation="volume_dropout",
                        quality=f"{quality:.3f}",
                        drop_fraction=effected.metadata.get("drop_fraction", "0.0"),
                        drop_percent=effected.metadata.get("drop_percent", "0.0"),
                        breakpoints=breakpoint_label,
                    )
            elif mode == "phone_style":
                if use_per_phoneme:
                    self.rendered_audio = augment_by_phoneme_segments(
                        self.current_audio,
                        transcript,
                        phoneme_qualities or [],
                        "phone_style",
                        noise_type=self.phone_style_noise_var.get(),
                    )
                else:
                    self.rendered_audio = apply_phone_style_augmentation(
                        self.current_audio,
                        effect_quality,
                        self.phone_style_noise_var.get(),
                        breakpoints,
                        transcript,
                        phoneme_qualities,
                    )
            elif mode == "pronunciation_safe":
                self.rendered_audio = render_pronunciation_safe(
                    self.current_audio,
                    transcript,
                    jibberish=max(0.0, min(1.0, float(self.jibberish_var.get()))),
                    clarity=max(0.0, min(1.0, float(self.clarity_var.get()))),
                    timing_instability=max(0.0, min(1.0, float(self.timing_instability_var.get()))),
                    allow_full_gibberish=bool(self.allow_full_gibberish_var.get()),
                )
            else:
                cached_sequence = self._get_cached_blabber_sequence()
                if use_per_phoneme:
                    if cached_sequence is None:
                        cached_sequence = self._compute_blabber_sequence()
                        self.cached_blabber_sequence = cached_sequence
                    self.rendered_audio = render_blabber_from_resolved_phones(
                        self.current_audio,
                        str(cached_sequence["transcript"]),
                        list(cached_sequence["source_phones"]),
                        list(cached_sequence["mutated_phones"]),
                        voice_mode=voice_mode,
                        speaker_name=male_speaker,
                        phoneme_render_backend=phoneme_render_backend,
                        backend_voice=backend_voice,
                        phoneme_qualities=phoneme_qualities or [],
                        segmentation="phonetic",
                        per_phone_preset_indices=(
                            list(cached_sequence["per_phone_preset_indices"])
                            if cached_sequence.get("per_phone_preset_indices") is not None
                            else None
                        ),
                        per_phone_distances=(
                            list(cached_sequence["per_phone_distances"])
                            if cached_sequence.get("per_phone_distances") is not None
                            else None
                        ),
                        per_phone_sequences=(
                            [list(sequence) for sequence in cached_sequence["per_phone_sequences"]]
                            if cached_sequence.get("per_phone_sequences") is not None
                            else None
                        ),
                        per_phone_numeric_entries_used=(
                            bool(cached_sequence["per_phone_numeric_entries_used"])
                            if cached_sequence.get("per_phone_numeric_entries_used") is not None
                            else None
                        ),
                    )
                    self.rendered_audio = apply_style_transfer(
                        self.rendered_audio,
                        style_transfer_backend,
                        target_voice=style_transfer_target_voice,
                        presets_path=self.style_transfer_presets_path,
                        conda_env_name=COQUI_ENV_NAME,
                        base_voice_mode=voice_mode,
                    )
                else:
                    if cached_sequence is None:
                        cached_sequence = self._compute_blabber_sequence()
                        self.cached_blabber_sequence = cached_sequence
                    resolved_breakpoints, breakpoint_label = resolve_breakpoints(
                        breakpoints,
                        self.current_audio,
                        str(cached_sequence["transcript"]),
                        "blabber",
                        quality,
                        source_phones=list(cached_sequence["source_phones"]),
                        mutated_phones=list(cached_sequence["mutated_phones"]),
                        phoneme_qualities=phoneme_qualities,
                    )
                    self.rendered_audio = render_blabber_from_resolved_phones(
                        self.current_audio,
                        str(cached_sequence["transcript"]),
                        list(cached_sequence["source_phones"]),
                        list(cached_sequence["mutated_phones"]),
                        voice_mode=voice_mode,
                        speaker_name=male_speaker,
                        phoneme_render_backend=phoneme_render_backend,
                        backend_voice=backend_voice,
                        quality=quality,
                        breakpoints=breakpoint_label,
                        segmentation="phonetic",
                        global_preset_index=(
                            int(cached_sequence["global_preset_index"])
                            if cached_sequence["global_preset_index"] is not None
                            else None
                        ),
                        expansion_used=bool(cached_sequence["expansion_used"]),
                        global_distance_total=(
                            float(cached_sequence["global_distance_total"])
                            if cached_sequence.get("global_distance_total") is not None
                            else None
                        ),
                        global_max_phoneme_distance=(
                            float(cached_sequence["global_max_phoneme_distance"])
                            if cached_sequence.get("global_max_phoneme_distance") is not None
                            else None
                        ),
                    )
                    self.rendered_audio = apply_style_transfer(
                        self.rendered_audio,
                        style_transfer_backend,
                        target_voice=style_transfer_target_voice,
                        presets_path=self.style_transfer_presets_path,
                        conda_env_name=COQUI_ENV_NAME,
                        base_voice_mode=voice_mode,
                    )
        except Exception as exc:
            self.rendered_audio = None
            tb = traceback.format_exc()
            self._set_output(f"Render failed\n\n{exc}\n\nTraceback:\n{tb}")
            messagebox.showerror("Render failed", str(exc))
            return

        metadata = self.rendered_audio.metadata
        details = [
            f"Rendered {mode}",
        ]
        if mode == "pronunciation_safe":
            details.append("Quality mode: pronunciation-safe sliders")
        else:
            details.append(f"Quality: {quality:.2f}" if not use_per_phoneme else "Quality: per-phoneme")
            details.append(f"Quality mode: {'per-phoneme' if use_per_phoneme else 'global'}")
        if mode == "volume_dropout":
            details.append(f"Dropped: {metadata.get('drop_percent', '0.0')}%")
        if "noise_type" in metadata:
            details.append(f"Noise: {metadata['noise_type']}")
        if mode == "blabber":
            details.append(f"Voice mode: {metadata.get('voice_mode', 'woman')}")
            if metadata.get("coqui_model_name"):
                details.append(f"Coqui model: {metadata['coqui_model_name']}")
            if metadata.get("coqui_speaker"):
                details.append(f"Coqui speaker: {metadata['coqui_speaker']}")
            if metadata.get("phoneme_render_backend"):
                details.append(f"Phoneme backend: {metadata['phoneme_render_backend']}")
            if metadata.get("phoneme_render_model_name"):
                details.append(f"Phoneme model: {metadata['phoneme_render_model_name']}")
            if metadata.get("kokoro_voice"):
                details.append(f"Kokoro voice: {metadata['kokoro_voice']}")
            details.append(f"Style transfer: {metadata.get('style_transfer_backend', 'none')}")
            if metadata.get("style_transfer_target_voice"):
                details.append(f"Style target: {metadata['style_transfer_target_voice']}")
            if metadata.get("style_transfer_status"):
                details.append(f"Style status: {metadata['style_transfer_status']}")
        if mode == "pronunciation_safe":
            details.append(f"Jibberish: {metadata.get('jibberish', '0.000')}")
            details.append(f"Clarity: {metadata.get('clarity', '0.000')}")
            details.append(f"Timing instability: {metadata.get('timing_instability', '0.000')}")
            details.append(
                "Full gibberish override: "
                + ("yes" if metadata.get("allow_full_gibberish") == "1" else "no")
            )
            if metadata.get("pronunciation_similarity_score"):
                details.append(
                    "Similarity: "
                    f"{metadata['pronunciation_similarity_score']} / {metadata.get('pronunciation_similarity_threshold', 'n/a')}"
                )
            if metadata.get("pronunciation_fragment_count"):
                details.append(f"Projected fragments: {metadata['pronunciation_fragment_count']}")
            if metadata.get("editor_name"):
                details.append(f"Editor: {metadata['editor_name']}")
            if metadata.get("assembler_name"):
                details.append(f"Assembler: {metadata['assembler_name']}")
            if metadata.get("pronunciation_selected_word_changed") == "1" and metadata.get("pronunciation_selected_word"):
                details.append(f"Selected word: {metadata['pronunciation_selected_word']}")
        if "breakpoints" in metadata:
            details.append(f"Breakpoints: {metadata['breakpoints']}")
        if "transcript" in metadata:
            details.append(f"Transcript: {metadata['transcript']}")
            details.append(f"Phones: {metadata.get('source_phones', '')}")
            details.append(f"Mutated: {metadata.get('substituted_phones', '')}")
            if metadata.get("global_preset_index") is not None:
                details.append(f"Global preset: {metadata['global_preset_index']}")
                if metadata.get("global_max_phoneme_distance") is not None:
                    details.append(f"Max global distance: {metadata['global_max_phoneme_distance']}")
                if metadata.get("global_distance_total") is not None:
                    details.append(f"Total distance: {metadata['global_distance_total']}")
                details.append(
                    "Expansion used: yes" if metadata.get("expansion_used") == "1" else "Expansion used: no"
                )
            if metadata.get("per_phone_preset_indices"):
                details.append(f"Per-phone presets: {metadata['per_phone_preset_indices']}")
            if metadata.get("per_phone_distances"):
                details.append(f"Per-phone distances: {metadata['per_phone_distances']}")
            if metadata.get("whole_word_distance_total"):
                details.append(f"Whole-word distance: {metadata['whole_word_distance_total']}")
            if metadata.get("per_phone_numeric_entries_used") is not None:
                details.append(
                    "Reference entries: numeric candidate distances"
                    if metadata.get("per_phone_numeric_entries_used") == "1"
                    else "Reference entries: legacy fallback ordering"
                )
            if metadata.get("per_phone_reference_warning") == "1":
                details.append("Reference warning: regenerate phoneme_distances.json for smoother per-phoneme ladders")
            if mode == "blabber" and self._get_cached_blabber_sequence() is not None:
                details.append("Sequence source: cached")
        elif self.detected_phones:
            details.append(f"Detected phones: {'-'.join(self.detected_phones)}")
        self._set_output("\n".join(details))

    def _preview(self) -> None:
        if self.rendered_audio is None:
            messagebox.showerror("No render", "Render an output first.")
            return
        preview_dir = Path(tempfile.gettempdir())
        self.preview_path = preview_dir / "speech_distortion_preview.wav"
        self.writer.write(str(self.preview_path), self.rendered_audio)
        try:
            import winsound

            winsound.PlaySound(str(self.preview_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._append_output(f"\nPreviewing {self.preview_path.name}")
        except Exception as exc:
            messagebox.showinfo(
                "Preview written",
                f"Preview WAV written to:\n{self.preview_path}\n\nPlayback failed: {exc}",
            )

    def _save_as(self) -> None:
        if self.rendered_audio is None:
            messagebox.showerror("No render", "Render an output first.")
            return
        source_name = Path(self.audio_path_var.get() or "output.wav").stem
        mode = self.mode_var.get()
        quality_token = "per_phoneme" if self.quality_mode_var.get() == "per_phoneme" else f"{self.quality_var.get():.2f}".replace(".", "_")
        output_path = filedialog.asksaveasfilename(
            title="Save distorted WAV",
            defaultextension=".wav",
            initialfile=f"{source_name}_{mode}_q{quality_token}.wav",
            filetypes=[("WAV files", "*.wav")],
        )
        if not output_path:
            return
        self.writer.write(output_path, self.rendered_audio)
        self._append_output(f"\nSaved to {output_path}")

    def _set_output(self, text: str) -> None:
        self.status_var.set(text)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.see(tk.END)

    def _append_output(self, text: str) -> None:
        current = self.output_text.get("1.0", tk.END).rstrip()
        combined = f"{current}{text}" if current else text
        self._set_output(combined)


def main() -> None:
    root = tk.Tk()
    app = SpeechDistortionGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
