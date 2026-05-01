import math

import numpy as np

from speech_distortion_gui import (
    MAX_DROPOUT_SILENCE_MS,
    apply_volume_dropout,
    slider_caption_for_mode,
    slider_value_text_for_mode,
    volume_dropout_fraction_from_quality,
)
from speech_distortion_pipeline.models import AudioBuffer


def build_audio(sample_count: int = 16000, sample_rate_hz: int = 16000) -> AudioBuffer:
    t = np.linspace(0.0, 1.0, num=sample_count, endpoint=False, dtype=np.float32)
    samples = (0.5 * np.sin(2.0 * math.pi * 220.0 * t)).astype(np.float32)
    return AudioBuffer(
        samples=samples.tolist(),
        sample_rate_hz=sample_rate_hz,
        channel_count=1,
        metadata={"source_path": "synthetic.wav"},
    )


def test_volume_dropout_fraction_mapping() -> None:
    assert volume_dropout_fraction_from_quality(1.0) == 0.0
    assert volume_dropout_fraction_from_quality(0.5) == 0.4
    assert volume_dropout_fraction_from_quality(0.0) == 0.8


def test_apply_volume_dropout_quality_one_preserves_audio() -> None:
    audio = build_audio()

    rendered = apply_volume_dropout(audio, 1.0)

    assert rendered.samples == audio.samples
    assert rendered.metadata["drop_fraction"] == "0.000"
    assert rendered.metadata["drop_percent"] == "0.0"


def test_apply_volume_dropout_caps_at_eighty_percent_drop_with_silence() -> None:
    audio = build_audio()

    rendered = apply_volume_dropout(audio, 0.0)
    rendered_samples = np.asarray(rendered.samples, dtype=np.float32)

    dropped_fraction = float(rendered.metadata["drop_fraction"])
    assert 0.75 <= dropped_fraction <= 0.80
    assert np.count_nonzero(rendered_samples == 0.0) > 0
    assert len(rendered.samples) == len(audio.samples)


def test_apply_volume_dropout_midpoint_targets_forty_percent_drop() -> None:
    audio = build_audio()

    rendered = apply_volume_dropout(audio, 0.5)

    dropped_fraction = float(rendered.metadata["drop_fraction"])
    assert 0.35 <= dropped_fraction <= 0.45
    assert rendered.metadata["drop_percent"] == f"{dropped_fraction * 100.0:.1f}"


def test_apply_volume_dropout_limits_silence_runs_to_two_hundred_ms() -> None:
    audio = build_audio(sample_count=32000, sample_rate_hz=16000)

    rendered = apply_volume_dropout(audio, 0.0)
    rendered_samples = np.asarray(rendered.samples, dtype=np.float32)
    zero_mask = rendered_samples == 0.0

    longest_run = 0
    current_run = 0
    for is_zero in zero_mask:
        if is_zero:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    max_allowed_samples = int(round((MAX_DROPOUT_SILENCE_MS / 1000.0) * audio.sample_rate_hz))
    assert longest_run <= max_allowed_samples


def test_apply_volume_dropout_is_deterministic() -> None:
    audio = build_audio()

    first = apply_volume_dropout(audio, 0.3)
    second = apply_volume_dropout(audio, 0.3)

    assert first.samples == second.samples
    assert first.metadata["drop_fraction"] == second.metadata["drop_fraction"]


def test_slider_text_switches_for_dropout_mode() -> None:
    assert slider_caption_for_mode("volume_dropout") == "Dropout"
    assert slider_caption_for_mode("static_noise") == "Quality"
    assert slider_value_text_for_mode("volume_dropout", 0.5) == "40%"
    assert slider_value_text_for_mode("static_noise", 0.5) == "0.50"
