from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from speech_distortion_pipeline.io import WavAudioReader, WavAudioWriter
from speech_distortion_pipeline.models import AudioBuffer
from speech_distortion_pipeline.resynthesis.fragment_synthesizer import resolve_python_command


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STYLE_TRANSFER_PRESETS_PATH = REPO_ROOT / "config" / "style_transfer_presets.json"
STYLE_TRANSFER_BACKEND_NONE = "none"
STYLE_TRANSFER_BACKEND_OPENVOICE_V2 = "openvoice_v2"
STYLE_TRANSFER_BACKENDS: tuple[str, str] = (
    STYLE_TRANSFER_BACKEND_NONE,
    STYLE_TRANSFER_BACKEND_OPENVOICE_V2,
)
STYLE_TRANSFER_MODEL_NAMES: dict[str, str] = {
    STYLE_TRANSFER_BACKEND_OPENVOICE_V2: "voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
}


@dataclass(frozen=True)
class StyleTransferPreset:
    name: str
    display_name: str
    target_gender: str
    target_wav_paths: tuple[str, ...]
    description: str = ""


class StyleTransferError(RuntimeError):
    """Raised when style transfer setup or conversion fails."""


def normalize_style_transfer_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in STYLE_TRANSFER_BACKENDS:
        raise ValueError(
            f"Unsupported style transfer backend '{value}'. Expected one of: {', '.join(STYLE_TRANSFER_BACKENDS)}."
        )
    return normalized


def style_transfer_model_name(backend: str) -> str:
    normalized = normalize_style_transfer_backend(backend)
    if normalized == STYLE_TRANSFER_BACKEND_NONE:
        return ""
    return STYLE_TRANSFER_MODEL_NAMES[normalized]


def load_style_transfer_presets(presets_path: str | Path | None = None) -> dict[str, StyleTransferPreset]:
    path = Path(presets_path) if presets_path is not None else DEFAULT_STYLE_TRANSFER_PRESETS_PATH
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_presets = payload.get("presets", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_presets, list):
        raise StyleTransferError(f"Style transfer preset file '{path}' must contain a top-level preset list.")

    parsed: dict[str, StyleTransferPreset] = {}
    for entry in raw_presets:
        preset = _parse_style_transfer_preset(entry, path)
        if preset is None:
            continue
        parsed[preset.name] = preset
    return parsed


def male_style_transfer_presets(presets_path: str | Path | None = None) -> list[StyleTransferPreset]:
    return [
        preset
        for preset in load_style_transfer_presets(presets_path).values()
        if preset.target_gender == "male"
    ]


def resolve_style_transfer_preset(
    name: str,
    presets_path: str | Path | None = None,
    required_gender: str | None = None,
) -> StyleTransferPreset:
    presets = load_style_transfer_presets(presets_path)
    normalized_name = str(name).strip()
    if not normalized_name:
        raise StyleTransferError("A style transfer target voice preset is required.")
    preset = presets.get(normalized_name)
    if preset is None:
        raise StyleTransferError(
            f"Style transfer preset '{normalized_name}' was not found in '{_preset_path_text(presets_path)}'."
        )
    if required_gender is not None and preset.target_gender != required_gender:
        raise StyleTransferError(
            f"Style transfer preset '{normalized_name}' has target gender '{preset.target_gender}', "
            f"expected '{required_gender}'."
        )
    if not preset.target_wav_paths:
        raise StyleTransferError(f"Style transfer preset '{normalized_name}' does not define any target WAV paths.")
    for wav_path in preset.target_wav_paths:
        if not Path(wav_path).is_file():
            raise StyleTransferError(
                f"Style transfer preset '{normalized_name}' references missing target WAV '{wav_path}'."
            )
    return preset


def apply_style_transfer(
    audio: AudioBuffer,
    backend: str,
    target_voice: str = "",
    presets_path: str | Path | None = None,
    conda_env_name: str = "coqui-blabber",
    base_voice_mode: str = "woman",
) -> AudioBuffer:
    normalized_backend = normalize_style_transfer_backend(backend)
    resolved_target_voice = str(target_voice).strip()
    if normalized_backend == STYLE_TRANSFER_BACKEND_NONE:
        return _clone_audio_with_metadata(
            audio,
            style_transfer_backend=STYLE_TRANSFER_BACKEND_NONE,
            style_transfer_target_voice="",
            style_transfer_status="not_requested",
            style_transfer_base_voice_mode=base_voice_mode,
        )

    preset = resolve_style_transfer_preset(
        resolved_target_voice,
        presets_path=presets_path,
        required_gender="male",
    )
    output_audio = _convert_with_coqui_vc(audio, normalized_backend, preset, conda_env_name=conda_env_name)
    return _clone_audio_with_metadata(
        output_audio,
        **dict(audio.metadata),
        style_transfer_backend=normalized_backend,
        style_transfer_target_voice=preset.name,
        style_transfer_status="success",
        style_transfer_base_voice_mode=base_voice_mode,
    )


def _preset_path_text(presets_path: str | Path | None) -> str:
    path = Path(presets_path) if presets_path is not None else DEFAULT_STYLE_TRANSFER_PRESETS_PATH
    return str(path)


def _parse_style_transfer_preset(entry: Any, manifest_path: Path) -> StyleTransferPreset | None:
    if not isinstance(entry, Mapping):
        return None
    raw_name = entry.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None

    display_name = entry.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = raw_name.strip()

    target_gender = str(entry.get("target_gender", "")).strip().lower() or "unknown"
    description = str(entry.get("description", "")).strip()
    target_wav_paths = _normalize_target_wav_paths(entry.get("target_wavs"), manifest_path)
    return StyleTransferPreset(
        name=raw_name.strip(),
        display_name=display_name.strip(),
        target_gender=target_gender,
        target_wav_paths=tuple(target_wav_paths),
        description=description,
    )


def _normalize_target_wav_paths(raw_value: Any, manifest_path: Path) -> list[str]:
    if isinstance(raw_value, str):
        candidates = [raw_value]
    elif isinstance(raw_value, list):
        candidates = [value for value in raw_value if isinstance(value, str)]
    else:
        return []

    resolved_paths: list[str] = []
    for value in candidates:
        stripped = value.strip()
        if not stripped:
            continue
        path = Path(stripped)
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        resolved_paths.append(str(path))
    return resolved_paths


def _clone_audio_with_metadata(audio: AudioBuffer, **metadata: str) -> AudioBuffer:
    merged = dict(audio.metadata)
    for key, value in metadata.items():
        merged[str(key)] = str(value)
    return AudioBuffer(
        samples=list(audio.samples),
        sample_rate_hz=audio.sample_rate_hz,
        channel_count=audio.channel_count,
        speaker_id=audio.speaker_id,
        metadata=merged,
    )


def _convert_with_coqui_vc(
    audio: AudioBuffer,
    backend: str,
    preset: StyleTransferPreset,
    conda_env_name: str,
) -> AudioBuffer:
    writer = WavAudioWriter()
    reader = WavAudioReader()
    with tempfile.TemporaryDirectory(prefix="speech_distortion_style_transfer_") as temp_dir:
        temp_root = Path(temp_dir)
        source_path = temp_root / "source.wav"
        output_path = temp_root / "converted.wav"
        writer.write(str(source_path), audio)

        _run_coqui_vc(
            source_wav_path=source_path,
            output_wav_path=output_path,
            backend=backend,
            target_wav_paths=preset.target_wav_paths,
            conda_env_name=conda_env_name,
        )

        if not output_path.is_file():
            raise StyleTransferError("Style transfer did not produce an output WAV.")
        converted = reader.read(str(output_path))
    return _resample_audio_buffer(converted, audio.sample_rate_hz)


def _run_coqui_vc(
    source_wav_path: Path,
    output_wav_path: Path,
    backend: str,
    target_wav_paths: Sequence[str],
    conda_env_name: str,
) -> None:
    fd_script, script_path = tempfile.mkstemp(prefix="speech_distortion_vc_", suffix=".py")
    os.close(fd_script)
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(_coqui_vc_script())
        env = os.environ.copy()
        env["COQUI_VC_MODEL"] = style_transfer_model_name(backend)
        env["COQUI_VC_SOURCE_WAV"] = str(source_wav_path)
        env["COQUI_VC_TARGET_WAVS"] = json.dumps(list(target_wav_paths))
        env["COQUI_VC_OUT"] = str(output_wav_path)
        env["COQUI_VC_USE_GPU"] = "0"
        completed = subprocess.run(
            resolve_python_command(conda_env_name) + [script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            details = stderr or stdout or f"Coqui VC subprocess exited with code {completed.returncode}."
            raise StyleTransferError(details)
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def _coqui_vc_script() -> str:
    return (
        "import json\n"
        "import os\n"
        "from TTS.api import TTS\n"
        "model_name = os.environ['COQUI_VC_MODEL']\n"
        "source_wav = os.environ['COQUI_VC_SOURCE_WAV']\n"
        "target_wavs = json.loads(os.environ['COQUI_VC_TARGET_WAVS'])\n"
        "out_path = os.environ['COQUI_VC_OUT']\n"
        "use_gpu = os.environ.get('COQUI_VC_USE_GPU', '0') == '1'\n"
        "tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)\n"
        "kwargs = {'source_wav': source_wav, 'file_path': out_path}\n"
        "if target_wavs:\n"
        "    kwargs['target_wav'] = target_wavs if len(target_wavs) > 1 else target_wavs[0]\n"
        "tts.voice_conversion_to_file(**kwargs)\n"
    )


def _resample_audio_buffer(audio: AudioBuffer, target_sample_rate_hz: int) -> AudioBuffer:
    if audio.sample_rate_hz == target_sample_rate_hz:
        return audio
    target_length = max(
        1,
        int(round(len(audio.samples) * float(target_sample_rate_hz) / float(audio.sample_rate_hz))),
    )
    resampled = _resample_samples(audio.samples, target_length)
    return AudioBuffer(
        samples=resampled,
        sample_rate_hz=target_sample_rate_hz,
        channel_count=audio.channel_count,
        speaker_id=audio.speaker_id,
        metadata=dict(audio.metadata),
    )


def _resample_samples(samples: Sequence[float], target_length: int) -> list[float]:
    if target_length <= 0:
        return []
    source = np.asarray(samples, dtype=np.float32)
    if source.size == 0:
        return [0.0] * target_length
    if source.size == target_length:
        return source.astype(np.float32, copy=True).tolist()
    if source.size == 1:
        return [float(source[0])] * target_length

    src_positions = np.linspace(0.0, 1.0, num=source.size, endpoint=True)
    dst_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(dst_positions, src_positions, source).astype(np.float32).tolist()
