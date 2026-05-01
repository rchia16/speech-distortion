from __future__ import annotations

import wave
from pathlib import Path

from speech_distortion_pipeline.models import AudioBuffer


class WavAudioReader:
    """Minimal PCM WAV reader for early pipeline slices."""

    def read(self, path: str) -> AudioBuffer:
        wav_path = Path(path)
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate_hz = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width == 1:
            integers = [byte - 128 for byte in raw]
            peak = 128.0
        elif sample_width == 2:
            integers = [
                int.from_bytes(raw[index : index + 2], byteorder="little", signed=True)
                for index in range(0, len(raw), 2)
            ]
            peak = 32768.0
        elif sample_width == 4:
            integers = [
                int.from_bytes(raw[index : index + 4], byteorder="little", signed=True)
                for index in range(0, len(raw), 4)
            ]
            peak = 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM width: {sample_width} bytes")

        if channel_count > 1:
            mono_samples = []
            for start in range(0, len(integers), channel_count):
                frame = integers[start : start + channel_count]
                mono_samples.append(sum(frame) / float(len(frame)))
            integers = mono_samples

        normalized = [sample / peak for sample in integers]
        return AudioBuffer(
            samples=normalized,
            sample_rate_hz=sample_rate_hz,
            channel_count=1,
            metadata={"source_path": str(wav_path)},
        )


class WavAudioWriter:
    """Minimal PCM WAV writer for bootstrap slices."""

    def write(self, path: str, audio: AudioBuffer) -> None:
        wav_path = Path(path)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = bytearray()
        for sample in audio.samples:
            clamped = max(-1.0, min(1.0, sample))
            value = int(round(clamped * 32767.0))
            pcm.extend(int(value).to_bytes(2, byteorder="little", signed=True))

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(audio.channel_count)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio.sample_rate_hz)
            wav_file.writeframes(bytes(pcm))
