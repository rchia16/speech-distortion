import importlib

import pytest


def _reload_blabber_config(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    for key in (
        "BLABBER_COQUI_MODEL",
        "BLABBER_COQUI_WOMAN_MODEL",
        "BLABBER_COQUI_MAN_MODEL",
        "BLABBER_COQUI_CLONE_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("speech_distortion_pipeline.phonology.blabber_config")
    return importlib.reload(module)


def test_resolve_blabber_voice_profile_defaults_woman_to_legacy_model(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _reload_blabber_config(monkeypatch)

    profile = config.resolve_blabber_voice_profile("woman")

    assert profile.voice_mode == "woman"
    assert profile.model_name == "tts_models/en/ljspeech/tacotron2-DDC_ph"
    assert profile.uses_source_speaker_wav is False
    assert profile.prephonemized is True
    assert profile.input_encoding == "ipa"


def test_resolve_blabber_voice_profile_uses_man_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _reload_blabber_config(
        monkeypatch,
        BLABBER_COQUI_MAN_MODEL="tts_models/en/vctk/vits",
    )

    profile = config.resolve_blabber_voice_profile("man")

    assert profile.voice_mode == "man"
    assert profile.model_name == "tts_models/en/vctk/vits"
    assert profile.uses_source_speaker_wav is False
    assert profile.prephonemized is True
    assert profile.input_encoding == "ipa"


def test_resolve_blabber_voice_profile_requires_man_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _reload_blabber_config(monkeypatch)

    with pytest.raises(ValueError, match="BLABBER_COQUI_MAN_MODEL is not set"):
        config.resolve_blabber_voice_profile("man")


def test_resolve_blabber_voice_profile_for_source_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _reload_blabber_config(monkeypatch)

    profile = config.resolve_blabber_voice_profile("source_clone")

    assert profile.voice_mode == "source_clone"
    assert profile.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert profile.uses_source_speaker_wav is True
    assert profile.prephonemized is False
    assert profile.language == "en"
    assert profile.input_encoding == "clone_text"


def test_normalize_blabber_voice_mode_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _reload_blabber_config(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported blabber voice mode"):
        config.normalize_blabber_voice_mode("robot")
