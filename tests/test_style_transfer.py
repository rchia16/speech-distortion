from __future__ import annotations

import json

from speech_distortion_pipeline.models import AudioBuffer
from speech_distortion_pipeline.resynthesis.style_transfer import (
    STYLE_TRANSFER_BACKEND_NONE,
    apply_style_transfer,
    load_style_transfer_presets,
    resolve_style_transfer_preset,
)


def test_load_style_transfer_presets_resolves_relative_paths(tmp_path):
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    first = reference_dir / "male_a.wav"
    second = reference_dir / "male_b.wav"
    first.write_bytes(b"")
    second.write_bytes(b"")
    manifest_path = tmp_path / "style_transfer_presets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "presets": [
                    {
                        "name": "male_demo",
                        "display_name": "Male Demo",
                        "target_gender": "male",
                        "target_wavs": [
                            "references/male_a.wav",
                            "references/male_b.wav",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    presets = load_style_transfer_presets(manifest_path)

    assert list(presets) == ["male_demo"]
    assert presets["male_demo"].target_wav_paths == (str(first.resolve()), str(second.resolve()))


def test_resolve_style_transfer_preset_requires_existing_name(tmp_path):
    manifest_path = tmp_path / "style_transfer_presets.json"
    manifest_path.write_text(json.dumps({"presets": []}), encoding="utf-8")

    try:
        resolve_style_transfer_preset("missing", manifest_path, required_gender="male")
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected missing preset resolution to fail.")


def test_apply_style_transfer_none_only_updates_metadata():
    audio = AudioBuffer(
        samples=[0.0, 0.1, -0.1],
        sample_rate_hz=16000,
        metadata={"voice_mode": "woman", "coqui_model_name": "tts_models/en/ljspeech/tacotron2-DDC_ph"},
    )

    updated = apply_style_transfer(
        audio,
        STYLE_TRANSFER_BACKEND_NONE,
        target_voice="",
        presets_path=None,
        conda_env_name="coqui-blabber",
        base_voice_mode="woman",
    )

    assert updated.samples == audio.samples
    assert updated.metadata["style_transfer_backend"] == "none"
    assert updated.metadata["style_transfer_target_voice"] == ""
    assert updated.metadata["style_transfer_status"] == "not_requested"
    assert updated.metadata["style_transfer_base_voice_mode"] == "woman"
