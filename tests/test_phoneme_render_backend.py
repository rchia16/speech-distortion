from __future__ import annotations

from speech_distortion_pipeline.resynthesis.phoneme_render_backend import (
    DEFAULT_KOKORO_VOICE,
    PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    PHONEME_RENDER_BACKEND_FASTPITCH,
    PHONEME_RENDER_BACKEND_KOKORO,
    PHONEME_RENDER_BACKEND_PHONEME_VITS,
    PhonemeRenderResult,
    backend_metadata,
    default_woman_phoneme_render_backend,
    normalize_kokoro_voice,
    normalize_phoneme_render_backend,
    probe_phoneme_render_backend,
)


def test_default_woman_backend_is_current_baseline():
    assert default_woman_phoneme_render_backend() == PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH


def test_normalize_phoneme_render_backend_accepts_known_values():
    assert normalize_phoneme_render_backend("FASTPITCH") == PHONEME_RENDER_BACKEND_FASTPITCH
    assert normalize_phoneme_render_backend("kokoro") == PHONEME_RENDER_BACKEND_KOKORO
    assert normalize_phoneme_render_backend("phoneme_vits") == PHONEME_RENDER_BACKEND_PHONEME_VITS


def test_nonconfigured_experimental_backend_probe_is_explicit():
    probe = probe_phoneme_render_backend(PHONEME_RENDER_BACKEND_FASTPITCH)

    assert probe.backend == PHONEME_RENDER_BACKEND_FASTPITCH
    assert probe.status in {"nemo_ready", "probe_failed", "external_configured"}


def test_kokoro_voice_alias_maps_to_real_backend_voice():
    requested, canonical = normalize_kokoro_voice("gm_fable")

    assert requested == "gm_fable"
    assert canonical == "bm_fable"


def test_kokoro_probe_is_explicit():
    probe = probe_phoneme_render_backend(PHONEME_RENDER_BACKEND_KOKORO)

    assert probe.backend == PHONEME_RENDER_BACKEND_KOKORO
    assert probe.status in {"kokoro_ready", "probe_failed"}


def test_backend_metadata_fields_are_stable():
    result = PhonemeRenderResult(
        samples=[],
        backend=PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
        model_name="tts_models/en/ljspeech/tacotron2-DDC_ph",
        input_encoding="ipa",
        prephonemized=True,
    )

    metadata = backend_metadata(result)

    assert metadata["phoneme_render_backend"] == PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH
    assert metadata["phoneme_render_model_name"] == "tts_models/en/ljspeech/tacotron2-DDC_ph"
    assert metadata["phoneme_render_input_encoding"] == "ipa"
    assert metadata["phoneme_render_prephonemized"] == "1"


def test_backend_metadata_includes_kokoro_voice_when_present():
    result = PhonemeRenderResult(
        samples=[],
        backend=PHONEME_RENDER_BACKEND_KOKORO,
        model_name="hexgrad/Kokoro-82M",
        input_encoding="pronunciation_markup",
        prephonemized=False,
        voice_name=DEFAULT_KOKORO_VOICE,
    )

    metadata = backend_metadata(result)

    assert metadata["kokoro_voice"] == DEFAULT_KOKORO_VOICE
