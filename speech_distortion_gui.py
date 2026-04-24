from __future__ import annotations

import math
import os
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
from speech_distortion_pipeline.phonology.g2p import HeuristicEnglishG2P
from speech_distortion_pipeline.resynthesis.fragment_synthesizer import (
    CoquiFragmentSynthesizer,
    CoquiSynthesisError,
    resolve_conda_command,
)
DEFAULT_INPUT = ROOT / "yes_slow.wav"
COQUI_ENV_NAME = os.environ.get("BLABBER_CONDA_ENV", "coqui-blabber")
COQUI_MODEL_NAME = os.environ.get(
    "BLABBER_COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph"
)
MAX_DROPOUT_FRACTION = 0.9


PHONE_SUBSTITUTIONS: dict[str, Sequence[str]] = {
    "R": ("W", "L"),
    "L": ("W", "Y"),
    "TH": ("F", "S"),
    "DH": ("D", "Z"),
    "S": ("SH", "T"),
    "Z": ("ZH", "D"),
    "SH": ("S", "CH"),
    "CH": ("SH", "T"),
    "JH": ("ZH", "D"),
    "K": ("T", "G"),
    "G": ("D", "K"),
    "T": ("K", "D"),
    "D": ("G", "T"),
    "P": ("B", "F"),
    "B": ("P", "M"),
    "F": ("TH", "P"),
    "V": ("DH", "B"),
    "ER": ("AH", "EH"),
    "IY": ("IH", "EY"),
    "IH": ("IY", "EH"),
    "EH": ("AE", "AH"),
    "AE": ("EH", "AH"),
    "AH": ("AE", "UH"),
    "OW": ("AW", "UH"),
    "UW": ("OW", "UH"),
    "AY": ("EY", "IY"),
    "EY": ("AY", "EH"),
    "OY": ("OW", "UH"),
    "AW": ("OW", "AH"),
    "W": ("Y", "L"),
    "Y": ("W", "IY"),
    "M": ("N", "B"),
    "N": ("M", "NG"),
    "NG": ("N", "G"),
}

MISMATCH_INSERTION_PHONES: Sequence[str] = ("HH", "Y", "W", "AH", "S", "R")

PHONE_TO_IPA: dict[str, str] = {
    "AE": "æ",
    "AH": "ʌ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

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
    frame_samples = max(1, int(round(sample_rate * lerp(0.012, 0.05, normalized))))
    frame_count = max(1, int(math.ceil(signal.shape[0] / float(frame_samples))))
    target_drop_frames = min(frame_count, max(0, int(round(frame_count * drop_fraction))))

    dropped_frames = np.zeros(frame_count, dtype=bool)
    remaining = target_drop_frames
    max_burst_frames = max(1, int(round(lerp(1.0, 6.0, normalized))))
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
    if severity <= 0.02:
        return list(phones)

    substituted = list(phones)
    candidate_indices = [index for index, phone in enumerate(phones) if phone in PHONE_SUBSTITUTIONS]
    if not candidate_indices:
        return substituted

    if phoneme_qualities is None:
        count = max(1, int(round(lerp(1.0, float(len(candidate_indices)), severity))))
        ordered = sorted(
            candidate_indices,
            key=lambda index: (
                phones[index] not in {"R", "L", "S", "TH", "K", "G"},
                index,
            ),
        )
        for offset, index in enumerate(ordered[:count]):
            choices = PHONE_SUBSTITUTIONS[phones[index]]
            choice_index = min(len(choices) - 1, int(math.floor(severity * len(choices))))
            substituted[index] = choices[(choice_index + offset) % len(choices)]
    else:
        for index in candidate_indices:
            local_severity = 1.0 - clamp_unit(float(phoneme_qualities[index]))
            if local_severity <= 0.02:
                continue
            choices = PHONE_SUBSTITUTIONS[phones[index]]
            choice_index = min(len(choices) - 1, int(math.floor(local_severity * len(choices))))
            substituted[index] = choices[choice_index]

    insertion_severity = severity
    if phoneme_qualities is not None and phoneme_qualities:
        insertion_severity = max((1.0 - clamp_unit(float(value)) for value in phoneme_qualities), default=0.0)

    if insertion_severity >= 0.3 and substituted:
        insertion_count = max(1, int(round(lerp(1.0, 3.0, insertion_severity))))
        for insertion_index in range(insertion_count):
            anchor = min(len(substituted), 1 + (insertion_index * 2))
            inserted_phone = "AH"
            if insertion_severity >= 0.55:
                inserted_phone = ("Y", "W", "HH")[insertion_index % 3]
            substituted.insert(anchor, inserted_phone)

    if insertion_severity >= 0.6 and len(substituted) >= 2:
        swap_index = min(len(substituted) - 2, max(0, len(substituted) // 2 - 1))
        substituted[swap_index], substituted[swap_index + 1] = (
            substituted[swap_index + 1],
            substituted[swap_index],
        )

    if insertion_severity >= 0.8 and substituted:
        substituted.append("AH")

    return substituted


def phones_to_ipa(phones: Sequence[str]) -> str:
    ipa = []
    for phone in phones:
        symbol = PHONE_TO_IPA.get(phone)
        if symbol is None:
            raise ValueError(f"No IPA mapping defined for phone '{phone}'.")
        ipa.append(symbol)
    return "".join(ipa)


def mutate_phone(phone: str, severity: float) -> str:
    severity = clamp_unit(severity)
    choices = PHONE_SUBSTITUTIONS.get(phone)
    if not choices or severity <= 0.02:
        return phone
    choice_index = min(len(choices) - 1, int(math.floor(severity * len(choices))))
    return choices[choice_index]


def build_per_phoneme_blabber_sequence(
    source_phones: Sequence[str],
    phoneme_qualities: Sequence[float],
) -> list[str]:
    mutated: list[str] = []
    for index, phone in enumerate(source_phones):
        local_quality = clamp_unit(float(phoneme_qualities[index])) if index < len(phoneme_qualities) else 1.0
        local_severity = 1.0 - local_quality
        primary = mutate_phone(phone, local_severity)
        mutated.append(primary)

        if local_severity < 0.3:
            continue

        substitutions = list(PHONE_SUBSTITUTIONS.get(phone, ()))
        insertion_pool: list[str] = []
        insertion_pool.extend(candidate for candidate in substitutions if candidate != primary)
        insertion_pool.extend(candidate for candidate in MISMATCH_INSERTION_PHONES if candidate != primary)

        if not insertion_pool:
            continue

        first_extra = insertion_pool[index % len(insertion_pool)]
        mutated.append(first_extra)

        if local_severity >= 0.7:
            second_extra = insertion_pool[(index + 2) % len(insertion_pool)]
            mutated.append(second_extra)

        if local_severity >= 0.9 and len(mutated) >= 3:
            mutated[-2], mutated[-1] = mutated[-1], mutated[-2]

    return mutated


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
) -> AudioBuffer:
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Blabber mode requires a single-word transcript.")
    if len(cleaned.split()) != 1:
        raise ValueError("Blabber mode currently supports exactly one word.")

    severity = 1.0 - quality
    if severity <= 0.001:
        return from_numpy(to_numpy(audio), audio, augmentation="blabber", quality=f"{quality:.3f}")

    ensure_coqui_backend(COQUI_ENV_NAME)

    g2p = HeuristicEnglishG2P()
    source_phones = g2p.phonemize_word(cleaned)
    if not source_phones:
        raise ValueError("Could not derive phones from the transcript.")

    substituted_phones = pick_substituted_phones(source_phones, severity, phoneme_qualities)
    synthesizer = CoquiFragmentSynthesizer(
        conda_env_name=COQUI_ENV_NAME,
        model_name=COQUI_MODEL_NAME,
    )
    synth_ipa = phones_to_ipa(substituted_phones)
    try:
        rendered = synthesizer._render_with_coqui(
            synth_ipa,
            audio.sample_rate_hz,
            prephonemized=True,
        )
    except CoquiSynthesisError as exc:
        raise RuntimeError(f"Coqui synthesis failed in conda env '{COQUI_ENV_NAME}'.\n\n{exc}") from exc
    if not rendered:
        raise RuntimeError(f"Coqui synthesis produced no audio in conda env '{COQUI_ENV_NAME}'.")

    signal = to_numpy(audio)
    natural_render = np.asarray(rendered, dtype=np.float32)
    natural_duration_sec = float(natural_render.shape[0]) / float(audio.sample_rate_hz)
    output = natural_render
    mutation_count = sum(
        1 for source_phone, mutated_phone in zip(source_phones, substituted_phones) if source_phone != mutated_phone
    )
    mutation_count += abs(len(substituted_phones) - len(source_phones))
    return from_numpy(
        output,
        audio,
        augmentation="blabber",
        quality=f"{quality:.3f}",
        transcript=cleaned,
        source_phones="-".join(source_phones),
        substituted_phones="-".join(substituted_phones),
        substituted_ipa=synth_ipa,
        mutation_count=str(mutation_count),
        natural_duration_sec=f"{natural_duration_sec:.4f}",
    )


def render_blabber_per_phoneme(
    audio: AudioBuffer,
    transcript: str,
    phoneme_qualities: Sequence[float],
) -> AudioBuffer:
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Blabber mode requires a single-word transcript.")
    if len(cleaned.split()) != 1:
        raise ValueError("Blabber mode currently supports exactly one word.")

    ensure_coqui_backend(COQUI_ENV_NAME)
    g2p = HeuristicEnglishG2P()
    source_phones = g2p.phonemize_word(cleaned)
    if not source_phones:
        raise ValueError("Could not derive phones from the transcript.")

    synthesizer = CoquiFragmentSynthesizer(
        conda_env_name=COQUI_ENV_NAME,
        model_name=COQUI_MODEL_NAME,
    )

    mutated_phones = build_per_phoneme_blabber_sequence(source_phones, phoneme_qualities)

    synth_ipa = phones_to_ipa(mutated_phones)
    try:
        rendered = synthesizer._render_with_coqui(
            synth_ipa,
            audio.sample_rate_hz,
            prephonemized=True,
        )
    except CoquiSynthesisError as exc:
        raise RuntimeError(f"Coqui synthesis failed in conda env '{COQUI_ENV_NAME}'.\n\n{exc}") from exc
    if not rendered:
        raise RuntimeError(f"Coqui synthesis produced no audio in conda env '{COQUI_ENV_NAME}'.")

    output = np.asarray(rendered, dtype=np.float32)
    return from_numpy(
        output,
        audio,
        augmentation="blabber",
        transcript=cleaned,
        source_phones="-".join(source_phones),
        substituted_phones="-".join(mutated_phones),
        substituted_ipa=synth_ipa,
        phoneme_qualities=",".join(f"{clamp_unit(float(v)):.2f}" for v in phoneme_qualities[: len(source_phones)]),
        mutation_count=str(
            sum(1 for src, dst in zip(source_phones, mutated_phones[: len(source_phones)]) if src != dst)
            + max(0, len(mutated_phones) - len(source_phones))
        ),
        segmentation="phonetic",
    )


def render_blabber_with_segments(
    audio: AudioBuffer,
    transcript: str,
    quality: float,
    breakpoints_str: str,
    phoneme_qualities: Sequence[float] | None = None,
) -> AudioBuffer:
    blabber = render_blabber(audio, transcript, quality, phoneme_qualities)
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


def ensure_coqui_backend(env_name: str) -> None:
    completed = subprocess.run(
        resolve_conda_command()
        + [
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('TTS') else 1)",
        ],
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


class SpeechDistortionGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Speech Distortion GUI")
        self.root.geometry("760x800")

        self.reader = WavAudioReader()
        self.writer = WavAudioWriter()
        self.current_audio: AudioBuffer | None = None
        self.rendered_audio: AudioBuffer | None = None
        self.preview_path: Path | None = None
        self.phoneme_quality_vars: list[tk.DoubleVar] = []
        self.detected_phones: list[str] = []

        self.audio_path_var = tk.StringVar(value=str(DEFAULT_INPUT if DEFAULT_INPUT.exists() else ""))
        self.mode_var = tk.StringVar(value="static_noise")
        self.phone_style_noise_var = tk.StringVar(value="white")
        self.quality_mode_var = tk.StringVar(value="global")
        self.quality_var = tk.DoubleVar(value=0.75)
        self.transcript_var = tk.StringVar(value="yes")
        self.breakpoints_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Choose a WAV file and render an augmentation.")
        self.subtitle_var = tk.StringVar(value=subtitle_text_for_mode(self.mode_var.get()))
        self.slider_label_var = tk.StringVar(value=slider_caption_for_mode(self.mode_var.get()))

        self._build_ui()
        self.transcript_var.trace_add("write", self._on_transcript_change)
        if self.audio_path_var.get():
            self._load_audio(Path(self.audio_path_var.get()))
        self._refresh_phoneme_controls()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

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

        info_frame = ttk.LabelFrame(container, text="Status", padding=12)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        self.output_text = ScrolledText(info_frame, wrap=tk.WORD, height=12)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.insert("1.0", self.status_var.get())

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(button_row, text="Render", command=self._render).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Preview", command=self._preview).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Save As", command=self._save_as).pack(side=tk.LEFT, padx=(8, 0))

        self._sync_quality_label()
        self._update_mode_state()
        self._update_quality_mode_state()

    def _sync_quality_label(self, *_args: object) -> None:
        self.quality_label.configure(
            text=slider_value_text_for_mode(self.mode_var.get(), self.quality_var.get())
        )

    def _on_transcript_change(self, *_args: object) -> None:
        self._refresh_phoneme_controls()

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
            scale.configure(command=lambda _value, var=quality_var, label=value_label: label.configure(text=f"{var.get():.2f}"))
        self._update_quality_mode_state()

    def _update_quality_mode_state(self) -> None:
        use_per_phoneme = self.quality_mode_var.get() == "per_phoneme"
        if use_per_phoneme:
            self.global_quality_scale.state(["disabled"])
        else:
            self.global_quality_scale.state(["!disabled"])
        for child in self.phoneme_controls_frame.winfo_children():
            for grandchild in child.winfo_children():
                if isinstance(grandchild, ttk.Scale):
                    if use_per_phoneme:
                        grandchild.state(["!disabled"])
                    else:
                        grandchild.state(["disabled"])

    def _update_mode_state(self) -> None:
        mode = self.mode_var.get()
        self.slider_label_var.set(slider_caption_for_mode(mode))
        self.subtitle_var.set(subtitle_text_for_mode(mode))
        self._sync_quality_label()
        self.transcript_entry.state(["!disabled"])
        if mode == "phone_style":
            self.phone_style_combo.state(["!disabled", "readonly"])
        else:
            self.phone_style_combo.state(["disabled"])

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
            self._set_output(
                f"Loaded {path.name}\n"
                f"Sample rate: {self.current_audio.sample_rate_hz} Hz\n"
                f"Duration: {duration:.2f} s"
            )
        except Exception as exc:
            self.current_audio = None
            self._set_output(f"Failed to load {path}: {exc}")

    def _render(self) -> None:
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
            else:
                if use_per_phoneme:
                    self.rendered_audio = render_blabber_per_phoneme(
                        self.current_audio,
                        transcript,
                        phoneme_qualities or [],
                    )
                else:
                    self.rendered_audio = render_blabber_with_segments(
                        self.current_audio,
                        transcript,
                        quality,
                        breakpoints,
                        phoneme_qualities,
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
            f"Quality: {quality:.2f}" if not use_per_phoneme else "Quality: per-phoneme",
        ]
        details.append(f"Quality mode: {'per-phoneme' if use_per_phoneme else 'global'}")
        if mode == "volume_dropout":
            details.append(f"Dropped: {metadata.get('drop_percent', '0.0')}%")
        if "noise_type" in metadata:
            details.append(f"Noise: {metadata['noise_type']}")
        if "breakpoints" in metadata:
            details.append(f"Breakpoints: {metadata['breakpoints']}")
        if "transcript" in metadata:
            details.append(f"Transcript: {metadata['transcript']}")
            details.append(f"Phones: {metadata.get('source_phones', '')}")
            details.append(f"Mutated: {metadata.get('substituted_phones', '')}")
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
