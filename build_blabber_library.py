#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_distortion_pipeline.io import WavAudioReader, WavAudioWriter
from speech_distortion_pipeline.models import AudioBuffer
from speech_distortion_pipeline.phonology.blabber_config import (
    COQUI_ENV_NAME,
    MISMATCH_INSERTION_PHONES,
    PHONE_SUBSTITUTIONS,
    PHONE_TO_CLONE_TEXT,
    PHONE_TO_IPA,
    resolve_blabber_voice_profile,
)
from speech_distortion_pipeline.phonology.g2p import HeuristicEnglishG2P
from speech_distortion_pipeline.phonology.phone_distance_reference import (
    bucketed_phone_choice,
    bucketed_phone_sequence,
    GLOBAL_BLABBER_PRESET_COUNT,
    quality_for_global_blabber_preset,
    ranked_neighbors_from_reference,
    resolve_per_phoneme_blabber_sequences,
    resolve_phone_blabber_sequence,
    resolve_soft_global_blabber_sequences,
)
from speech_distortion_pipeline.resynthesis.fragment_synthesizer import (
    CoquiFragmentSynthesizer,
    CoquiSynthesisError,
    resolve_python_command,
)
from speech_distortion_pipeline.resynthesis.phoneme_render_backend import (
    DEFAULT_KOKORO_VOICE,
    PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    PHONEME_RENDER_BACKEND_FASTPITCH,
    PHONEME_RENDER_BACKEND_KOKORO,
    PHONEME_RENDER_BACKEND_PHONEME_VITS,
    PHONEME_RENDER_BACKENDS,
    KOKORO_VOICE_OPTIONS,
    backend_metadata,
    default_woman_phoneme_render_backend,
    probe_phoneme_render_backend,
    render_with_phoneme_backend,
)
from speech_distortion_pipeline.resynthesis.style_transfer import (
    DEFAULT_STYLE_TRANSFER_PRESETS_PATH,
    STYLE_TRANSFER_BACKENDS,
    STYLE_TRANSFER_BACKEND_NONE,
    apply_style_transfer,
)

LIBRARY_BLABBER_VOICE_MODE = "woman"
GENERATION_MODE_BOTH = "both"
ASSET_BANK_GLOBAL = "global"
ASSET_BANK_PER_PHONEME = "per_phoneme"
DEFAULT_LIBRARY_PHONEME_RENDER_BACKEND = PHONEME_RENDER_BACKEND_KOKORO
GENDER_FEMALE = "female"
GENDER_MALE = "male"
GENDER_CHOICES = (GENDER_FEMALE, GENDER_MALE)


@dataclass
class ManifestEntry:
    audio_path: Path
    transcript: str
    phoneme_count: int


@dataclass(frozen=True)
class BuildTask:
    entry: ManifestEntry
    generation_mode: str
    output_root: Path
    overwrite: bool
    gender: str
    global_max_phoneme_distance: float | None
    phoneme_render_backend: str
    backend_voice: str | None
    style_transfer_backend: str
    style_transfer_target_voice: str
    style_transfer_presets_path: Path
    global_preset_count: int
    phoneme_step_size: float
    use_gpu: bool


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_gender(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in GENDER_CHOICES:
        raise ValueError(f"Gender must be one of: {', '.join(GENDER_CHOICES)}.")
    return normalized


def gender_to_voice_mode(gender: str) -> str:
    return "woman" if normalize_gender(gender) == GENDER_FEMALE else "man"


def infer_kokoro_voice_gender(voice_name: str) -> str:
    canonical = str(voice_name).strip().lower()
    if len(canonical) < 2 or canonical[1] not in {"f", "m"}:
        raise ValueError(f"Could not infer Kokoro voice gender from '{voice_name}'.")
    return GENDER_FEMALE if canonical[1] == "f" else GENDER_MALE


def clamp(samples: np.ndarray) -> np.ndarray:
    return np.clip(samples, -1.0, 1.0)


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


def _select_boundary_indices(scores: np.ndarray, count: int, min_gap: int) -> list[int]:
    if count <= 0 or scores.size == 0:
        return []
    ranked = list(np.argsort(scores)[::-1])
    selected: list[int] = []
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
) -> list[int]:
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
    phones: Sequence[str],
) -> list[tuple[int, int, str]]:
    if not phones:
        return []
    total_samples = len(audio.samples)
    if total_samples <= 0:
        return []

    boundaries = _audio_segment_boundaries(
        to_numpy(audio),
        audio.sample_rate_hz,
        len(phones),
    )
    spans: list[tuple[int, int, str]] = []
    for index, phone in enumerate(phones):
        start = boundaries[index]
        end = boundaries[index + 1] if index + 1 < len(boundaries) else total_samples
        spans.append((start, max(start + 1, min(total_samples, end)), phone))
    if spans:
        spans[-1] = (spans[-1][0], total_samples, spans[-1][2])
    return spans


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
    candidate_sequences, preset_indices, distances, total_distance, expansion_used, numeric_entries_used = resolve_per_phoneme_blabber_sequences(
        source_phones,
        [clamp_unit(float(value)) for value in phoneme_qualities],
        fallback_map=PHONE_SUBSTITUTIONS,
    )
    mutated = [phone for sequence in candidate_sequences for phone in sequence]
    return mutated, preset_indices, distances, total_distance, expansion_used, candidate_sequences, numeric_entries_used


def resolve_global_blabber_phone_sequence(
    transcript: str,
    quality: float,
    max_phoneme_distance: float | None = None,
) -> tuple[str, list[str], list[str], int, bool, float]:
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Transcript is required.")
    if len(cleaned.split()) != 1:
        raise ValueError("Only single-word transcripts are supported.")

    g2p = HeuristicEnglishG2P()
    source_phones = g2p.phonemize_word(cleaned)
    if not source_phones:
        raise ValueError("Could not derive phones from the transcript.")

    candidate_sequences, preset_index, expansion_used, total_distance = resolve_soft_global_blabber_sequences(
        source_phones,
        quality,
        fallback_map=PHONE_SUBSTITUTIONS,
        max_phoneme_distance=max_phoneme_distance,
    )
    mutated_phones = [phone for sequence in candidate_sequences for phone in sequence]
    return cleaned, source_phones, mutated_phones, preset_index, expansion_used, total_distance


def phones_to_ipa(phones: Sequence[str]) -> str:
    try:
        return "".join(PHONE_TO_IPA[phone] for phone in phones)
    except KeyError as exc:
        raise ValueError(f"No IPA mapping defined for phone '{exc.args[0]}'.") from exc


def phones_to_clone_text(phones: Sequence[str]) -> str:
    try:
        return "".join(PHONE_TO_CLONE_TEXT[phone] for phone in phones)
    except KeyError as exc:
        raise ValueError(f"No clone-text mapping defined for phone '{exc.args[0]}'.") from exc


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
        raise RuntimeError(
            f"Coqui TTS is not available in conda env '{env_name}'. "
            f"Install it there before generating blabber assets."
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


def render_blabber_sequence(
    audio: AudioBuffer,
    phones: Sequence[str],
    voice_mode: str,
    use_gpu: bool = False,
) -> tuple[np.ndarray, str, dict[str, str]]:
    profile = resolve_blabber_voice_profile(voice_mode)
    if profile.uses_source_speaker_wav or not profile.prephonemized or profile.input_encoding != "ipa":
        synthesizer = CoquiFragmentSynthesizer(
            conda_env_name=COQUI_ENV_NAME,
            model_name=profile.model_name,
            speaker_wav_path=resolve_source_speaker_wav_path(audio, True) if profile.uses_source_speaker_wav else None,
            use_gpu=use_gpu,
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

    result = render_with_phoneme_backend(
        audio,
        phones,
        backend=default_woman_phoneme_render_backend(),
        conda_env_name=COQUI_ENV_NAME,
    )
    return np.asarray(result.samples, dtype=np.float32), result.model_name, backend_metadata(result)


def render_blabber_per_phoneme(
    audio: AudioBuffer,
    transcript: str,
    phoneme_qualities: Sequence[float],
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
    style_transfer_backend: str = STYLE_TRANSFER_BACKEND_NONE,
    style_transfer_target_voice: str = "",
    style_transfer_presets_path: str | Path | None = None,
    use_gpu: bool = False,
) -> tuple[AudioBuffer, list[str], list[str]]:
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Transcript is required.")
    if len(cleaned.split()) != 1:
        raise ValueError("Only single-word transcripts are supported.")

    g2p = HeuristicEnglishG2P()
    source_phones = g2p.phonemize_word(cleaned)
    if not source_phones:
        raise ValueError("Could not derive phones from the transcript.")
    if len(phoneme_qualities) != len(source_phones):
        raise ValueError(
            f"Transcript '{cleaned}' resolves to {len(source_phones)} phonemes "
            f"({', '.join(source_phones)}), but {len(phoneme_qualities)} values were provided."
        )

    (
        mutated_phones,
        per_phone_preset_indices,
        per_phone_distances,
        total_distance,
        expansion_used,
        per_phone_sequences,
        per_phone_numeric_entries_used,
    ) = build_per_phoneme_blabber_sequence(source_phones, phoneme_qualities)
    try:
        if phoneme_render_backend == PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH:
            output, coqui_model_name, render_backend_meta = render_blabber_sequence(
                audio,
                mutated_phones,
                LIBRARY_BLABBER_VOICE_MODE,
                use_gpu=use_gpu,
            )
        else:
            result = render_with_phoneme_backend(
                audio,
                mutated_phones,
                backend=phoneme_render_backend,
                conda_env_name=COQUI_ENV_NAME,
                backend_voice=backend_voice,
            )
            output = np.asarray(result.samples, dtype=np.float32)
            coqui_model_name = result.model_name
            render_backend_meta = backend_metadata(result)
    except (CoquiSynthesisError, RuntimeError) as exc:
        raise RuntimeError(f"Phoneme render failed in conda env '{COQUI_ENV_NAME}'.\n\n{exc}") from exc
    rendered_audio = from_numpy(
        output,
        audio,
        augmentation="blabber",
        transcript=cleaned,
        source_phones="-".join(source_phones),
        substituted_phones="-".join(mutated_phones),
        substituted_ipa=phones_to_ipa(mutated_phones),
        phoneme_qualities=",".join(format_quality_token(value) for value in phoneme_qualities),
        per_phone_preset_indices=",".join(str(int(value)) for value in per_phone_preset_indices),
        per_phone_distances=",".join(f"{float(value):.6f}" for value in per_phone_distances),
        per_phone_distance_total=f"{total_distance:.6f}",
        whole_word_distance_total=f"{total_distance:.6f}",
        per_phone_sequences=";".join("-".join(sequence) for sequence in per_phone_sequences),
        expansion_used="1" if expansion_used else "0",
        per_phone_numeric_entries_used="1" if per_phone_numeric_entries_used else "0",
        per_phone_reference_warning="0" if per_phone_numeric_entries_used else "1",
        voice_mode=LIBRARY_BLABBER_VOICE_MODE,
        coqui_model_name=coqui_model_name,
        source_speaker_wav="0",
        style_transfer_backend="none",
        style_transfer_target_voice="",
        style_transfer_status="not_requested",
        **render_backend_meta,
    )
    rendered_audio = apply_style_transfer(
        rendered_audio,
        style_transfer_backend,
        target_voice=style_transfer_target_voice,
        presets_path=style_transfer_presets_path,
        conda_env_name=COQUI_ENV_NAME,
        base_voice_mode=LIBRARY_BLABBER_VOICE_MODE,
        use_gpu=use_gpu,
    )
    return rendered_audio, source_phones, mutated_phones


def render_blabber_global(
    audio: AudioBuffer,
    transcript: str,
    quality: float,
    max_phoneme_distance: float | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
    style_transfer_backend: str = STYLE_TRANSFER_BACKEND_NONE,
    style_transfer_target_voice: str = "",
    style_transfer_presets_path: str | Path | None = None,
    use_gpu: bool = False,
) -> tuple[AudioBuffer, list[str], list[str], int, bool, float]:
    cleaned, source_phones, mutated_phones, preset_index, expansion_used, total_distance = resolve_global_blabber_phone_sequence(
        transcript,
        quality,
        max_phoneme_distance=max_phoneme_distance,
    )
    try:
        if phoneme_render_backend == PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH:
            output, coqui_model_name, render_backend_meta = render_blabber_sequence(
                audio,
                mutated_phones,
                LIBRARY_BLABBER_VOICE_MODE,
                use_gpu=use_gpu,
            )
        else:
            result = render_with_phoneme_backend(
                audio,
                mutated_phones,
                backend=phoneme_render_backend,
                conda_env_name=COQUI_ENV_NAME,
                backend_voice=backend_voice,
            )
            output = np.asarray(result.samples, dtype=np.float32)
            coqui_model_name = result.model_name
            render_backend_meta = backend_metadata(result)
    except (CoquiSynthesisError, RuntimeError) as exc:
        raise RuntimeError(f"Phoneme render failed in conda env '{COQUI_ENV_NAME}'.\n\n{exc}") from exc
    metadata = {
        "augmentation": "blabber",
        "quality": f"{quality:.3f}",
        "transcript": cleaned,
        "source_phones": "-".join(source_phones),
        "substituted_phones": "-".join(mutated_phones),
        "substituted_ipa": phones_to_ipa(mutated_phones),
        "global_preset_index": str(preset_index),
        "expansion_used": "1" if expansion_used else "0",
        "global_distance_total": f"{total_distance:.6f}",
        "voice_mode": LIBRARY_BLABBER_VOICE_MODE,
        "coqui_model_name": coqui_model_name,
        "source_speaker_wav": "0",
        "global_progression_policy": "deterministic_distance_ladder",
        "style_transfer_backend": "none",
        "style_transfer_target_voice": "",
        "style_transfer_status": "not_requested",
    }
    metadata.update(render_backend_meta)
    if max_phoneme_distance is not None:
        metadata["global_max_phoneme_distance"] = f"{max_phoneme_distance:.3f}"
    rendered_audio = from_numpy(output, audio, **metadata)
    rendered_audio = apply_style_transfer(
        rendered_audio,
        style_transfer_backend,
        target_voice=style_transfer_target_voice,
        presets_path=style_transfer_presets_path,
        conda_env_name=COQUI_ENV_NAME,
        base_voice_mode=LIBRARY_BLABBER_VOICE_MODE,
        use_gpu=use_gpu,
    )
    return rendered_audio, source_phones, mutated_phones, preset_index, expansion_used, total_distance


def format_quality_token(value: float) -> str:
    clamped = clamp_unit(float(value))
    text = f"{clamped:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def safe_word_token(word: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in word).strip("_") or "word"


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def validate_builder_gender_configuration(
    gender: str,
    phoneme_render_backend: str,
    backend_voice: str | None,
) -> None:
    normalized_gender = normalize_gender(gender)
    normalized_backend = str(phoneme_render_backend).strip().lower()
    if normalized_backend != PHONEME_RENDER_BACKEND_KOKORO:
        return
    if not backend_voice:
        raise ValueError("Kokoro generation requires an explicit backend voice.")
    voice_gender = infer_kokoro_voice_gender(backend_voice)
    if voice_gender != normalized_gender:
        raise ValueError(
            f"Kokoro voice '{backend_voice}' is {voice_gender}, but '--gender {normalized_gender}' was requested."
        )


def finalize_rendered_audio_gender(
    rendered_audio: AudioBuffer,
    gender: str,
) -> AudioBuffer:
    normalized_gender = normalize_gender(gender)
    rendered_audio.metadata["gender"] = normalized_gender
    rendered_audio.metadata["voice_mode"] = gender_to_voice_mode(normalized_gender)
    return rendered_audio


def normalize_builder_generation_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {GENERATION_MODE_BOTH, "global_soft", ASSET_BANK_PER_PHONEME}:
        raise ValueError(
            f"Unsupported generation mode '{value}'. Expected one of: {GENERATION_MODE_BOTH}, global_soft, {ASSET_BANK_PER_PHONEME}."
        )
    return normalized


def normalize_library_backend_voice(backend: str, backend_voice: str | None) -> str | None:
    normalized_backend = str(backend).strip().lower()
    if normalized_backend != PHONEME_RENDER_BACKEND_KOKORO:
        return None
    requested = (backend_voice or DEFAULT_KOKORO_VOICE).strip().lower()
    if requested not in KOKORO_VOICE_OPTIONS:
        raise ValueError(
            f"Unsupported Kokoro voice '{backend_voice}'. Expected one of: {', '.join(KOKORO_VOICE_OPTIONS)}."
        )
    return requested


def manifest_row_to_entry(row: dict[str, Any], manifest_path: Path) -> ManifestEntry:
    audio_value = row.get("audio_path") or row.get("file") or row.get("path")
    transcript = str(row.get("transcript") or row.get("word") or "").strip()
    raw_phoneme_count = row.get("phoneme_count") or row.get("num_phonemes") or row.get("number_of_phonemes")

    if not audio_value:
        raise ValueError("Each manifest row must include 'audio_path'.")
    if not transcript:
        raise ValueError("Each manifest row must include 'transcript'.")
    if raw_phoneme_count is None:
        raise ValueError("Each manifest row must include 'phoneme_count'.")
    phoneme_count = int(raw_phoneme_count)
    if phoneme_count <= 0:
        raise ValueError("'phoneme_count' must be a positive integer.")

    audio_path = Path(str(audio_value))
    if not audio_path.is_absolute():
        audio_path = (manifest_path.parent / audio_path).resolve()
    return ManifestEntry(
        audio_path=audio_path,
        transcript=transcript,
        phoneme_count=phoneme_count,
    )


def load_manifest(path: Path) -> list[ManifestEntry]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [manifest_row_to_entry(row, path) for row in reader]
    if suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return [manifest_row_to_entry(row, path) for row in reader]
    if suffix == ".jsonl":
        entries: list[ManifestEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from exc
                entries.append(manifest_row_to_entry(row, path))
        return entries
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must contain a top-level array.")
        return [manifest_row_to_entry(row, path) for row in payload]
    raise ValueError("Manifest must be .csv, .tsv, .json, or .jsonl")


def spans_to_json(
    spans: Sequence[tuple[int, int, str]],
    sample_rate_hz: int,
) -> list[dict[str, Any]]:
    return [
        {
            "phone": phone,
            "start_sample": start,
            "end_sample": end,
            "start_sec": round(start / float(sample_rate_hz), 6),
            "end_sec": round(end / float(sample_rate_hz), 6),
            "duration_sec": round((end - start) / float(sample_rate_hz), 6),
        }
        for start, end, phone in spans
    ]


def build_sidecar_payload(
    input_audio_path: Path,
    output_audio_path: Path,
    transcript: str,
    generation_mode: str,
    phoneme_values: Sequence[float],
    source_audio: AudioBuffer,
    rendered_audio: AudioBuffer,
    source_phones: Sequence[str],
    mutated_phones: Sequence[str],
) -> dict[str, Any]:
    source_spans = derive_phone_sample_spans(source_audio, source_phones)
    output_spans = derive_phone_sample_spans(rendered_audio, mutated_phones)
    whole_word_distance_total = rendered_audio.metadata.get("whole_word_distance_total")
    payload = {
        "input_audio_path": str(input_audio_path),
        "output_audio_path": str(output_audio_path),
        "transcript": transcript,
        "word": transcript,
        "gender": rendered_audio.metadata.get("gender"),
        "generation_mode": generation_mode,
        "phoneme_values": [clamp_unit(float(value)) for value in phoneme_values],
        "source_phoneme_count": len(source_phones),
        "output_phoneme_count": len(mutated_phones),
        "source_phonemes": list(source_phones),
        "output_phonemes": list(mutated_phones),
        "source_alignment": spans_to_json(source_spans, source_audio.sample_rate_hz),
        "output_alignment": spans_to_json(output_spans, rendered_audio.sample_rate_hz),
        "sample_rate_hz": rendered_audio.sample_rate_hz,
        "duration_sec": round(len(rendered_audio.samples) / float(rendered_audio.sample_rate_hz), 6),
        "style_transfer": {
            "backend": rendered_audio.metadata.get("style_transfer_backend", "none"),
            "target_voice": rendered_audio.metadata.get("style_transfer_target_voice", ""),
            "status": rendered_audio.metadata.get("style_transfer_status", "not_requested"),
            "base_voice_mode": rendered_audio.metadata.get(
                "style_transfer_base_voice_mode",
                rendered_audio.metadata.get("voice_mode", LIBRARY_BLABBER_VOICE_MODE),
            ),
        },
        "metadata": dict(rendered_audio.metadata),
    }
    if whole_word_distance_total:
        payload["whole_word_distance_total"] = float(whole_word_distance_total)
    return payload


def generate_phoneme_value_sets(
    phoneme_count: int,
    step_size: float = 0.1,
) -> list[list[float]]:
    normalized_step_size = float(step_size)
    if normalized_step_size <= 0.0 or normalized_step_size > 1.0:
        raise ValueError("'phoneme_step_size' must be > 0.0 and <= 1.0.")

    step_count = int(round(1.0 / normalized_step_size))
    if not np.isclose(step_count * normalized_step_size, 1.0, atol=1e-8):
        raise ValueError("'phoneme_step_size' must divide 1.0 evenly, for example 0.5, 0.25, 0.2, or 0.1.")

    available_values = [index * normalized_step_size for index in range(step_count + 1)]
    if phoneme_count > len(available_values):
        raise ValueError(
            f"Cannot generate non-repeating 0.0-1.0 step-{normalized_step_size:g} permutations for {phoneme_count} phonemes. "
            f"Maximum supported phoneme_count is {len(available_values)}."
        )

    generated = [
        [float(value) for value in combo]
        for combo in itertools.permutations(available_values, phoneme_count)
    ]
    baseline = [1.0] * phoneme_count
    if baseline not in generated:
        generated.append(baseline)
    return generated


def generate_global_quality_presets(
    preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
) -> list[tuple[int, float]]:
    # Generate a fixed soft-to-moderate global ladder. The resolver normalizes each preset by word length
    # and delays 2-phone expansions until the strongest preset, which keeps longer words noticeably softer.
    return [
        (preset_index, quality_for_global_blabber_preset(preset_index, preset_count=preset_count))
        for preset_index in range(preset_count)
    ]


def process_entry(
    entry: ManifestEntry,
    output_root: Path,
    overwrite: bool,
    gender: str,
    reader: WavAudioReader,
    writer: WavAudioWriter,
    generation_mode: str,
    global_max_phoneme_distance: float | None = None,
    phoneme_render_backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    backend_voice: str | None = None,
    style_transfer_backend: str = STYLE_TRANSFER_BACKEND_NONE,
    style_transfer_target_voice: str = "",
    style_transfer_presets_path: str | Path | None = None,
    global_preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
    phoneme_step_size: float = 0.1,
    use_gpu: bool = False,
) -> dict[str, Any]:
    normalized_gender = normalize_gender(gender)
    validate_builder_gender_configuration(normalized_gender, phoneme_render_backend, backend_voice)
    if not entry.audio_path.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {entry.audio_path}")

    audio = reader.read(str(entry.audio_path))
    detected_phones = HeuristicEnglishG2P().phonemize_word(entry.transcript.strip())
    if len(detected_phones) != entry.phoneme_count:
        raise ValueError(
            f"Manifest phoneme_count={entry.phoneme_count} for '{entry.transcript}' does not match "
            f"detected phoneme count={len(detected_phones)} ({', '.join(detected_phones)})."
        )

    generated_items: list[dict[str, Any]] = []
    stem = safe_word_token(entry.transcript)
    if generation_mode == "global_soft":
        word_output_dir = output_root / ASSET_BANK_GLOBAL / stem
        word_output_dir.mkdir(parents=True, exist_ok=True)
        for preset_index, quality in generate_global_quality_presets(preset_count=global_preset_count):
            (
                rendered_audio,
                source_phones,
                mutated_phones,
                resolved_preset_index,
                expansion_used,
                total_distance,
            ) = render_blabber_global(
                audio,
                entry.transcript,
                quality,
                max_phoneme_distance=global_max_phoneme_distance,
                phoneme_render_backend=phoneme_render_backend,
                backend_voice=backend_voice,
                style_transfer_backend=style_transfer_backend,
                style_transfer_target_voice=style_transfer_target_voice,
                style_transfer_presets_path=style_transfer_presets_path,
                use_gpu=use_gpu,
            )
            rendered_audio = finalize_rendered_audio_gender(rendered_audio, normalized_gender)

            output_audio_path = word_output_dir / (
                f"{stem}_global_p{preset_index:02d}_q{format_quality_token(quality)}.wav"
            )
            output_json_path = output_audio_path.with_suffix(".json")

            if not overwrite and (output_audio_path.exists() or output_json_path.exists()):
                raise FileExistsError(
                    f"Refusing to overwrite existing output '{output_audio_path.name}'. Use --overwrite."
                )

            writer.write(str(output_audio_path), rendered_audio)
            payload = build_sidecar_payload(
                entry.audio_path,
                output_audio_path,
                entry.transcript.strip(),
                ASSET_BANK_GLOBAL,
                [quality],
                audio,
                rendered_audio,
                source_phones,
                mutated_phones,
            )
            output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            generated_items.append(
                {
                    "transcript": entry.transcript.strip(),
                    "word": entry.transcript.strip(),
                    "gender": normalized_gender,
                    "audio_path": relative_to_root(output_audio_path, output_root),
                    "json_path": relative_to_root(output_json_path, output_root),
                    "sidecar_path": relative_to_root(output_json_path, output_root),
                    "source_phonemes": source_phones,
                    "output_phonemes": mutated_phones,
                    "generation_mode": ASSET_BANK_GLOBAL,
                    "quality": quality,
                    "phoneme_values": [quality],
                    "voice_mode": rendered_audio.metadata.get("voice_mode", gender_to_voice_mode(normalized_gender)),
                    "global_preset_index": resolved_preset_index,
                    "expansion_used": expansion_used,
                    "global_distance_total": total_distance,
                    "global_max_phoneme_distance": global_max_phoneme_distance,
                    "style_transfer_backend": rendered_audio.metadata.get("style_transfer_backend", "none"),
                    "style_transfer_target_voice": rendered_audio.metadata.get("style_transfer_target_voice", ""),
                    "phoneme_render_backend": rendered_audio.metadata.get("phoneme_render_backend", ""),
                    "phoneme_render_model_name": rendered_audio.metadata.get("phoneme_render_model_name", ""),
                }
            )
    else:
        word_output_dir = output_root / ASSET_BANK_PER_PHONEME / stem
        word_output_dir.mkdir(parents=True, exist_ok=True)
        for phoneme_values in generate_phoneme_value_sets(entry.phoneme_count, step_size=phoneme_step_size):
            rendered_audio, source_phones, mutated_phones = render_blabber_per_phoneme(
                audio,
                entry.transcript,
                phoneme_values,
                phoneme_render_backend=phoneme_render_backend,
                backend_voice=backend_voice,
                style_transfer_backend=style_transfer_backend,
                style_transfer_target_voice=style_transfer_target_voice,
                style_transfer_presets_path=style_transfer_presets_path,
                use_gpu=use_gpu,
            )
            rendered_audio = finalize_rendered_audio_gender(rendered_audio, normalized_gender)

            quality_suffix = "_".join(format_quality_token(value) for value in phoneme_values)
            output_audio_path = word_output_dir / f"{stem}_per_phoneme_{quality_suffix}.wav"
            output_json_path = output_audio_path.with_suffix(".json")

            if not overwrite and (output_audio_path.exists() or output_json_path.exists()):
                raise FileExistsError(
                    f"Refusing to overwrite existing output '{output_audio_path.name}'. Use --overwrite."
                )

            writer.write(str(output_audio_path), rendered_audio)
            payload = build_sidecar_payload(
                entry.audio_path,
                output_audio_path,
                entry.transcript.strip(),
                ASSET_BANK_PER_PHONEME,
                phoneme_values,
                audio,
                rendered_audio,
                source_phones,
                mutated_phones,
            )
            output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            generated_items.append(
                {
                    "transcript": entry.transcript.strip(),
                    "word": entry.transcript.strip(),
                    "gender": normalized_gender,
                    "audio_path": relative_to_root(output_audio_path, output_root),
                    "json_path": relative_to_root(output_json_path, output_root),
                    "sidecar_path": relative_to_root(output_json_path, output_root),
                    "source_phonemes": source_phones,
                    "output_phonemes": mutated_phones,
                    "generation_mode": ASSET_BANK_PER_PHONEME,
                    "voice_mode": rendered_audio.metadata.get("voice_mode", gender_to_voice_mode(normalized_gender)),
                    "phoneme_values": phoneme_values,
                    "whole_word_distance_total": float(rendered_audio.metadata["whole_word_distance_total"]),
                    "style_transfer_backend": rendered_audio.metadata.get("style_transfer_backend", "none"),
                    "style_transfer_target_voice": rendered_audio.metadata.get("style_transfer_target_voice", ""),
                    "phoneme_render_backend": rendered_audio.metadata.get("phoneme_render_backend", ""),
                    "phoneme_render_model_name": rendered_audio.metadata.get("phoneme_render_model_name", ""),
                }
            )

    return {
        "transcript": entry.transcript.strip(),
        "word": entry.transcript.strip(),
        "generation_mode": ASSET_BANK_GLOBAL if generation_mode == "global_soft" else ASSET_BANK_PER_PHONEME,
        "generated_count": len(generated_items),
        "items": generated_items,
    }


def run_build_task(task: BuildTask) -> dict[str, Any]:
    reader = WavAudioReader()
    writer = WavAudioWriter()
    return process_entry(
        task.entry,
        task.output_root,
        task.overwrite,
        task.gender,
        reader,
        writer,
        task.generation_mode,
        global_max_phoneme_distance=task.global_max_phoneme_distance,
        phoneme_render_backend=task.phoneme_render_backend,
        backend_voice=task.backend_voice,
        style_transfer_backend=task.style_transfer_backend,
        style_transfer_target_voice=task.style_transfer_target_voice,
        style_transfer_presets_path=task.style_transfer_presets_path,
        global_preset_count=task.global_preset_count,
        phoneme_step_size=task.phoneme_step_size,
        use_gpu=task.use_gpu,
    )


def build_root_asset_index(generated: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in generated:
        for asset in item.get("items", []):
            row = dict(asset)
            row.setdefault("transcript", item.get("transcript", ""))
            row.setdefault("word", item.get("word", item.get("transcript", "")))
            rows.append(row)
    return rows


def load_existing_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    return [dict(item) for item in payload if isinstance(item, dict)]


def asset_index_identity(row: dict[str, Any]) -> tuple[str, str]:
    preferred_path = (
        row.get("sidecar_path")
        or row.get("json_path")
        or row.get("audio_path")
        or row.get("output_audio_path")
        or ""
    )
    generation_mode = str(row.get("generation_mode") or "")
    return (str(preferred_path), generation_mode)


def merge_root_asset_index(
    existing_rows: Sequence[dict[str, Any]],
    new_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {
        asset_index_identity(row): dict(row) for row in existing_rows
    }
    for row in new_rows:
        merged[asset_index_identity(row)] = dict(row)
    return list(merged.values())


def build_summary_groups_from_index(
    rows: Sequence[dict[str, Any]],
    generation_mode: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("generation_mode") != generation_mode:
            continue
        transcript = str(row.get("transcript") or row.get("word") or "").strip()
        if transcript not in grouped:
            grouped[transcript] = {
                "transcript": transcript,
                "word": str(row.get("word") or transcript),
                "generation_mode": generation_mode,
                "generated_count": 0,
                "items": [],
            }
        grouped[transcript]["items"].append(dict(row))

    summaries: list[dict[str, Any]] = []
    for transcript in sorted(grouped):
        item = grouped[transcript]
        item["items"].sort(
            key=lambda entry: str(
                entry.get("sidecar_path")
                or entry.get("json_path")
                or entry.get("audio_path")
                or ""
            )
        )
        item["generated_count"] = len(item["items"])
        summaries.append(item)
    return summaries


def filter_generated_by_mode(generated: Sequence[dict[str, Any]], generation_mode: str) -> list[dict[str, Any]]:
    return [dict(item) for item in generated if item.get("generation_mode") == generation_mode]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate standalone blabber phoneme-mismatch assets from a manifest of "
            "audio files, one-word transcripts, and phoneme counts."
        )
    )
    parser.add_argument("--manifest", required=True, help="Path to a .csv, .tsv, .json, or .jsonl manifest.")
    parser.add_argument(
        "--output-dir",
        default="speech-assets",
        help="Directory where generated wav/json assets will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs with the same generated filenames.",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print the generated asset list as JSON instead of plain text.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=(GENERATION_MODE_BOTH, "global_soft", "per_phoneme"),
        default=GENERATION_MODE_BOTH,
        help="Generate both banks by default, or restrict output to global_soft or per_phoneme.",
    )
    parser.add_argument(
        "--gender",
        choices=GENDER_CHOICES,
        default=GENDER_FEMALE,
        help="Authoritative output gender metadata for generated assets.",
    )
    parser.add_argument(
        "--global-max-phoneme-distance",
        type=float,
        default=None,
        help=(
            "Optional quality-0 word-distance endpoint for global_soft generation. "
            "Individual candidates above this distance are excluded."
        ),
    )
    parser.add_argument(
        "--global-preset-count",
        type=int,
        default=GLOBAL_BLABBER_PRESET_COUNT,
        help="Number of global quality presets to export for global_soft generation.",
    )
    parser.add_argument(
        "--phoneme-step-size",
        type=float,
        default=0.1,
        help="Step size for per-phoneme quality values between 0.0 and 1.0. Must divide 1.0 evenly.",
    )
    parser.add_argument(
        "--phoneme-render-backend",
        choices=PHONEME_RENDER_BACKENDS,
        default=DEFAULT_LIBRARY_PHONEME_RENDER_BACKEND,
        help="Direct phoneme render backend for woman Blabber generation.",
    )
    parser.add_argument(
        "--kokoro-voice",
        choices=KOKORO_VOICE_OPTIONS,
        default=DEFAULT_KOKORO_VOICE,
        help="Kokoro voice used when --phoneme-render-backend=kokoro.",
    )
    parser.add_argument(
        "--style-transfer-backend",
        choices=STYLE_TRANSFER_BACKENDS,
        default=STYLE_TRANSFER_BACKEND_NONE,
        help="Optional post-render style transfer backend to apply after the woman Blabber render.",
    )
    parser.add_argument(
        "--style-transfer-target-voice",
        default="",
        help="Named target voice preset from the style transfer preset manifest.",
    )
    parser.add_argument(
        "--style-transfer-presets",
        default=str(DEFAULT_STYLE_TRANSFER_PRESETS_PATH),
        help="Path to the JSON style transfer preset manifest.",
    )
    parser.add_argument(
        "--print-backend-probes",
        action="store_true",
        help="Print phoneme-render backend compatibility probes as JSON and exit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of manifest-entry generation tasks to run in parallel. Defaults to 1.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable existing Coqui GPU paths for synthesis and style transfer when available.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))
    use_gpu = bool(args.use_gpu)
    gender = normalize_gender(str(args.gender))
    global_preset_count = int(args.global_preset_count)
    phoneme_step_size = float(args.phoneme_step_size)
    if use_gpu and jobs > 1:
        print("GPU mode enabled; forcing --jobs=1 to avoid GPU oversubscription.", file=sys.stderr)
        jobs = 1
    if global_preset_count <= 0:
        raise ValueError("'--global-preset-count' must be a positive integer.")

    normalized_generation_mode = normalize_builder_generation_mode(str(args.generation_mode))
    backend_voice = normalize_library_backend_voice(str(args.phoneme_render_backend), str(args.kokoro_voice))
    if (
        str(args.phoneme_render_backend) in {
            PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
            PHONEME_RENDER_BACKEND_PHONEME_VITS,
        }
        or str(args.style_transfer_backend) != STYLE_TRANSFER_BACKEND_NONE
    ):
        ensure_coqui_backend(COQUI_ENV_NAME)
    if args.print_backend_probes:
        probes = [
            {
                "backend": probe.backend,
                "ok": probe.ok,
                "status": probe.status,
                "detail": probe.detail,
                "model_name": probe.model_name,
            }
            for probe in (
                probe_phoneme_render_backend(PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH, COQUI_ENV_NAME),
                probe_phoneme_render_backend(PHONEME_RENDER_BACKEND_FASTPITCH, COQUI_ENV_NAME),
                probe_phoneme_render_backend(PHONEME_RENDER_BACKEND_KOKORO, COQUI_ENV_NAME),
            )
        ]
        print(json.dumps(probes, indent=2))
        return 0
    entries = load_manifest(manifest_path)
    if not entries:
        raise ValueError("Manifest did not contain any entries.")

    selected_modes = (
        ("global_soft", "per_phoneme")
        if normalized_generation_mode == GENERATION_MODE_BOTH
        else (normalized_generation_mode,)
    )
    tasks = [
        BuildTask(
            entry=entry,
            generation_mode=selected_mode,
            output_root=output_dir,
            overwrite=bool(args.overwrite),
            gender=gender,
            global_max_phoneme_distance=(
                max(0.0, float(args.global_max_phoneme_distance))
                if args.global_max_phoneme_distance is not None
                else None
            ),
            phoneme_render_backend=str(args.phoneme_render_backend),
            backend_voice=backend_voice,
            style_transfer_backend=str(args.style_transfer_backend),
            style_transfer_target_voice=str(args.style_transfer_target_voice),
            style_transfer_presets_path=Path(args.style_transfer_presets).resolve(),
            global_preset_count=global_preset_count,
            phoneme_step_size=phoneme_step_size,
            use_gpu=use_gpu,
        )
        for entry in entries
        for selected_mode in selected_modes
    ]
    generated: list[dict[str, Any]] = []
    if jobs == 1:
        generated = [run_build_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_task = {
                executor.submit(run_build_task, task): (task_index, task)
                for task_index, task in enumerate(tasks)
            }
            task_results: list[tuple[int, dict[str, Any]]] = []
            for future in concurrent.futures.as_completed(future_to_task):
                task_index, task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed generating assets for transcript='{task.entry.transcript}' mode='{task.generation_mode}'."
                    ) from exc
                task_results.append((task_index, result))
            generated = [result for _, result in sorted(task_results, key=lambda item: item[0])]

    index_path = output_dir / "blabber_asset_index.json"
    existing_root_index = load_existing_json_array(index_path)
    current_root_index = build_root_asset_index(generated)
    root_index = merge_root_asset_index(existing_root_index, current_root_index)
    index_path.write_text(json.dumps(root_index, indent=2), encoding="utf-8")
    (output_dir / "global_summary.json").write_text(
        json.dumps(build_summary_groups_from_index(root_index, ASSET_BANK_GLOBAL), indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_phoneme_summary.json").write_text(
        json.dumps(build_summary_groups_from_index(root_index, ASSET_BANK_PER_PHONEME), indent=2),
        encoding="utf-8",
    )

    if args.print_summary_json:
        print(json.dumps(generated, indent=2))
    else:
        for item in generated:
            print(f"{item['transcript']}: generated {item['generated_count']} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
