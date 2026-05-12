from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from speech_distortion_pipeline.models import AudioBuffer, AudioSegment, EditPlan, EditType, PhoneGraph, PhoneNode, TimingPlan


def resolve_conda_command() -> list[str]:
    """Resolve a Windows-safe conda invocation for subprocess calls."""
    conda_exe = os.environ.get("CONDA_EXE")
    candidates = [
        conda_exe,
        shutil.which("conda"),
        shutil.which("conda.bat"),
        shutil.which("conda.exe"),
    ]
    conda_path = next((candidate for candidate in candidates if candidate), None)
    if not conda_path:
        return ["conda"]

    lowered = conda_path.lower()
    if lowered.endswith(".bat") or lowered.endswith(".cmd"):
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c", conda_path]
    return [conda_path]


def resolve_python_command(target_conda_env_name: str | None = None) -> list[str]:
    """Use the current interpreter when already inside the requested env, else fall back to conda run."""
    current_prefix = Path(sys.prefix).name.strip().lower()
    target_name = str(target_conda_env_name or "").strip().lower()
    if target_name and current_prefix == target_name:
        return [sys.executable]
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    if target_name and current_env == target_name:
        return [sys.executable]
    current_prefix_path = os.environ.get("CONDA_PREFIX", "").strip()
    if not current_env and not current_prefix_path:
        return [sys.executable]
    conda_command = resolve_conda_command()
    if conda_command == ["conda"]:
        return [sys.executable]
    return conda_command + ["run", "-n", target_conda_env_name or current_env or "base", "python"]


class FragmentSynthesizer(Protocol):
    """Synthesizes replacement fragments for additions and substitutions."""

    def synthesize(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> list[AudioSegment]:
        raise NotImplementedError


@dataclass
class HeuristicFragmentSynthesizer:
    """Dependency-free fragment synthesizer for substitutions and additions."""

    synthesizer_name: str = "heuristic_fragment_synthesizer_v1"

    _PHONE_FREQS: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._PHONE_FREQS is None:
            self._PHONE_FREQS = {
                "AH": 520.0,
                "AE": 660.0,
                "EH": 590.0,
                "IH": 700.0,
                "IY": 760.0,
                "OW": 430.0,
                "UW": 360.0,
                "AY": 610.0,
                "EY": 640.0,
                "OY": 500.0,
                "AW": 470.0,
                "ER": 540.0,
                "R": 300.0,
                "L": 340.0,
                "W": 280.0,
                "Y": 420.0,
                "M": 240.0,
                "N": 260.0,
                "NG": 220.0,
                "B": 190.0,
                "D": 210.0,
                "G": 170.0,
                "P": 0.0,
                "T": 0.0,
                "K": 0.0,
                "S": 0.0,
                "Z": 0.0,
                "SH": 0.0,
                "TH": 0.0,
                "DH": 0.0,
                "F": 0.0,
                "V": 0.0,
                "CH": 0.0,
                "JH": 0.0,
                "HH": 0.0,
            }

    def synthesize(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> list[AudioSegment]:
        node_map = {node.index: node for node in graph.nodes}
        budget_map = self._budget_by_target(plan, timing)
        fragments: List[AudioSegment] = []

        for operation in plan.operations:
            if operation.edit_type not in {EditType.SUBSTITUTE, EditType.DISTORTED_SUBSTITUTE, EditType.ADD}:
                continue
            if not operation.target_phone_indices:
                continue

            target_index = operation.target_phone_indices[0]
            target_node = node_map.get(target_index)
            if target_node is None:
                continue

            start_sec, end_sec = self._fragment_timing(target_node, operation, budget_map.get(target_index))
            phones = self._phones_for_operation(target_node, operation)
            samples = self._render_phone_sequence(
                phones=phones,
                sample_rate_hz=audio.sample_rate_hz,
                duration_sec=max(0.03, end_sec - start_sec),
                distorted=(operation.edit_type == EditType.DISTORTED_SUBSTITUTE),
                seed=target_index,
            )
            fragments.append(
                AudioSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    samples=samples,
                    label="{0}:{1}".format(operation.edit_type.value, "-".join(phones)),
                    source=self.synthesizer_name,
                )
            )

        return fragments

    def _budget_by_target(
        self, plan: EditPlan, timing: TimingPlan
    ) -> Dict[int, Tuple[float, float]]:
        lookup: Dict[int, Tuple[float, float]] = {}
        lengthening_ops = [
            operation
            for operation in plan.operations
            if operation.edit_type in {EditType.LENGTHEN_VOWEL, EditType.LENGTHEN_CONSONANT}
            and operation.target_phone_indices
        ]
        for budget in timing.budgets:
            for operation in lengthening_ops:
                target = operation.target_phone_indices[0]
                if budget.window_start_phone_index <= target <= budget.window_end_phone_index:
                    lookup[target] = (budget.requested_delta_sec, budget.compensated_delta_sec)
        return lookup

    def _fragment_timing(
        self,
        target_node: PhoneNode,
        operation,
        budget: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        start_sec = float(target_node.start_sec or 0.0)
        end_sec = float(target_node.end_sec or start_sec + 0.06)
        base_duration = max(0.03, end_sec - start_sec)
        extra = 0.0
        if operation.edit_type == EditType.ADD:
            extra += base_duration * max(1, len(operation.inserted_phones)) * 0.35
        if budget is not None:
            extra += budget[1]
        return start_sec, start_sec + base_duration + extra

    def _phones_for_operation(self, target_node: PhoneNode, operation) -> List[str]:
        if operation.edit_type == EditType.ADD:
            phones = list(operation.inserted_phones) + [target_node.phone]
            return phones or [target_node.phone]
        if operation.replacement_phones:
            return list(operation.replacement_phones)
        return [target_node.phone]

    def _render_phone_sequence(
        self,
        phones: List[str],
        sample_rate_hz: int,
        duration_sec: float,
        distorted: bool,
        seed: int,
    ) -> List[float]:
        total_samples = max(1, int(round(duration_sec * sample_rate_hz)))
        per_phone = max(1, total_samples // max(1, len(phones)))
        rendered: List[float] = []
        rng = random.Random(seed)
        for index, phone in enumerate(phones):
            length = per_phone if index < len(phones) - 1 else max(1, total_samples - len(rendered))
            rendered.extend(self._render_phone(phone, length, sample_rate_hz, distorted, rng))
        if len(rendered) != total_samples:
            rendered = rendered[:total_samples] if len(rendered) > total_samples else rendered + [0.0] * (total_samples - len(rendered))
        return rendered

    def _render_phone(
        self,
        phone: str,
        length: int,
        sample_rate_hz: int,
        distorted: bool,
        rng: random.Random,
    ) -> List[float]:
        if length <= 0:
            return []
        if self._is_noise_phone(phone):
            return self._noise_phone(phone, length, distorted, rng)
        freq = self._PHONE_FREQS.get(phone, 440.0)
        if freq <= 0.0:
            return self._burst_phone(length, distorted, rng)
        return self._tonal_phone(freq, length, sample_rate_hz, distorted)

    def _tonal_phone(
        self, freq: float, length: int, sample_rate_hz: int, distorted: bool
    ) -> List[float]:
        samples: List[float] = []
        for index in range(length):
            t = float(index) / float(sample_rate_hz)
            envelope = math.sin(math.pi * (float(index) / float(max(length - 1, 1))))
            value = (
                0.45 * math.sin(2.0 * math.pi * freq * t)
                + 0.20 * math.sin(2.0 * math.pi * freq * 2.01 * t)
                + 0.08 * math.sin(2.0 * math.pi * freq * 3.97 * t)
            ) * envelope
            if distorted:
                value += 0.06 * math.sin(2.0 * math.pi * freq * 6.1 * t)
                value *= 0.85 if (index // 13) % 2 == 0 else 0.55
            samples.append(self._clamp(value))
        return samples

    def _noise_phone(
        self, phone: str, length: int, distorted: bool, rng: random.Random
    ) -> List[float]:
        scale = 0.22 if phone in {"S", "SH", "F", "TH"} else 0.18
        output: List[float] = []
        state = 0.0
        for index in range(length):
            noise = (rng.random() * 2.0) - 1.0
            state = (0.82 * state) + (0.18 * noise)
            envelope = math.sin(math.pi * (float(index) / float(max(length - 1, 1))))
            value = state * scale * envelope
            if distorted:
                value *= 1.35 if (index // 7) % 2 == 0 else 0.7
            output.append(self._clamp(value))
        return output

    def _burst_phone(self, length: int, distorted: bool, rng: random.Random) -> List[float]:
        closure = max(1, int(length * 0.45))
        output = [0.0] * closure
        burst_len = max(1, length - closure)
        for index in range(burst_len):
            envelope = 1.0 - (float(index) / float(max(burst_len, 1)))
            value = (((rng.random() * 2.0) - 1.0) * 0.35 * envelope)
            if distorted:
                value *= 0.8 if index % 2 == 0 else 1.25
            output.append(self._clamp(value))
        return output[:length]

    def _is_noise_phone(self, phone: str) -> bool:
        return phone in {"S", "Z", "SH", "TH", "DH", "F", "V", "HH", "CH", "JH"}

    def _clamp(self, value: float) -> float:
        return max(-1.0, min(1.0, value))


@dataclass
class SayFragmentSynthesizer:
    """Real speech fragment backend using macOS `say`."""

    synthesizer_name: str = "system_say_fragment_synthesizer_v1"
    voice: str = "Samantha"
    rate_wpm: int = 220

    _PHONE_TEXT: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._PHONE_TEXT is None:
            self._PHONE_TEXT = {
                "AH": "uh",
                "AE": "a",
                "EH": "eh",
                "IH": "ih",
                "IY": "ee",
                "OW": "oh",
                "UW": "oo",
                "AY": "eye",
                "EY": "ay",
                "OY": "oy",
                "AW": "ow",
                "ER": "er",
                "R": "ruh",
                "L": "luh",
                "W": "wuh",
                "Y": "yuh",
                "M": "muh",
                "N": "nuh",
                "NG": "ng",
                "B": "buh",
                "D": "duh",
                "G": "guh",
                "P": "puh",
                "T": "tuh",
                "K": "kuh",
                "S": "sss",
                "Z": "zzz",
                "SH": "shh",
                "TH": "thh",
                "DH": "thuh",
                "F": "fff",
                "V": "vvv",
                "CH": "chuh",
                "JH": "juh",
                "HH": "huh",
            }

    def synthesize(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> list[AudioSegment]:
        helper = HeuristicFragmentSynthesizer()
        node_map = {node.index: node for node in graph.nodes}
        budget_map = helper._budget_by_target(plan, timing)
        fragments: List[AudioSegment] = []

        for operation in plan.operations:
            if operation.edit_type not in {EditType.SUBSTITUTE, EditType.DISTORTED_SUBSTITUTE, EditType.ADD}:
                continue
            if not operation.target_phone_indices:
                continue
            target_index = operation.target_phone_indices[0]
            target_node = node_map.get(target_index)
            if target_node is None:
                continue

            start_sec, end_sec = helper._fragment_timing(target_node, operation, budget_map.get(target_index))
            phones = helper._phones_for_operation(target_node, operation)
            text = self._phones_to_text(phones, operation.edit_type)
            samples = self._render_speech(text, audio.sample_rate_hz)
            if not samples:
                samples = helper._render_phone_sequence(
                    phones=phones,
                    sample_rate_hz=audio.sample_rate_hz,
                    duration_sec=max(0.03, end_sec - start_sec),
                    distorted=(operation.edit_type == EditType.DISTORTED_SUBSTITUTE),
                    seed=target_index,
                )

            target_samples = max(1, int(round(max(0.03, end_sec - start_sec) * audio.sample_rate_hz)))
            samples = self._resample(samples, target_samples)
            if operation.edit_type == EditType.DISTORTED_SUBSTITUTE:
                samples = self._distort(samples)

            fragments.append(
                AudioSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    samples=samples,
                    label="{0}:{1}".format(operation.edit_type.value, "-".join(phones)),
                    source=self.synthesizer_name,
                )
            )
        return fragments

    def _phones_to_text(self, phones: List[str], edit_type: EditType) -> str:
        pieces = [self._PHONE_TEXT.get(phone, phone.lower()) for phone in phones]
        if edit_type == EditType.ADD:
            return " ".join(pieces)
        if len(pieces) == 1:
            return pieces[0]
        return "".join(pieces)

    def _render_speech(self, text: str, target_sample_rate_hz: int) -> List[float]:
        fd_aiff, path_aiff = tempfile.mkstemp(suffix=".aiff")
        os.close(fd_aiff)
        fd_wav, path_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd_wav)
        try:
            completed = subprocess.run(
                ["say", "-v", self.voice, "-r", str(self.rate_wpm), "-o", path_aiff, text],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                return []
            converted = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@44100", path_aiff, path_wav],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if converted.returncode != 0:
                return []
            samples, source_rate = self._read_wav(path_wav)
            if not samples:
                return []
            if source_rate != target_sample_rate_hz:
                samples = self._resample(samples, int(round(len(samples) * float(target_sample_rate_hz) / float(source_rate))))
            return self._trim_silence(samples)
        finally:
            if os.path.exists(path_aiff):
                os.unlink(path_aiff)
            if os.path.exists(path_wav):
                os.unlink(path_wav)

    def _read_wav(self, path: str) -> Tuple[List[float], int]:
        with wave.open(path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width != 2:
            return [], frame_rate
        integers = [
            int.from_bytes(raw[index : index + 2], byteorder="little", signed=True)
            for index in range(0, len(raw), 2)
        ]
        if channel_count > 1:
            mono: List[float] = []
            for start in range(0, len(integers), channel_count):
                frame = integers[start : start + channel_count]
                mono.append(sum(frame) / float(len(frame)))
            integers = mono
        return ([float(value) / 32768.0 for value in integers], frame_rate)

    def _trim_silence(self, samples: List[float]) -> List[float]:
        if not samples:
            return []
        threshold = 0.01
        start = 0
        end = len(samples)
        while start < len(samples) and abs(samples[start]) < threshold:
            start += 1
        while end > start and abs(samples[end - 1]) < threshold:
            end -= 1
        trimmed = samples[start:end]
        return trimmed if trimmed else samples

    def _resample(self, samples: List[float], target_length: int) -> List[float]:
        if target_length <= 0:
            return []
        if not samples:
            return [0.0] * target_length
        if len(samples) == target_length:
            return list(samples)
        if len(samples) == 1:
            return [samples[0]] * target_length
        output: List[float] = []
        scale = float(len(samples) - 1) / float(max(target_length - 1, 1))
        for index in range(target_length):
            position = index * scale
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            mix = position - left
            output.append((samples[left] * (1.0 - mix)) + (samples[right] * mix))
        return output

    def _distort(self, samples: List[float]) -> List[float]:
        distorted: List[float] = []
        for index, sample in enumerate(samples):
            scaled = sample * (0.8 if (index // 11) % 2 == 0 else 0.55)
            distorted.append(max(-1.0, min(1.0, scaled)))
        return distorted


@dataclass
class CoquiFragmentSynthesizer:
    """Speech fragment backend using coqui-tts inside a conda environment."""

    synthesizer_name: str = "coqui_tts_fragment_synthesizer_v1"
    conda_env_name: str = "torch"
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    speaker_wav_path: Optional[str] = None
    speaker_name: Optional[str] = None
    use_gpu: bool = False

    def synthesize(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> list[AudioSegment]:
        helper = HeuristicFragmentSynthesizer()
        fallback_say = SayFragmentSynthesizer()
        node_map = {node.index: node for node in graph.nodes}
        budget_map = helper._budget_by_target(plan, timing)
        fragments: List[AudioSegment] = []

        for operation in plan.operations:
            if operation.edit_type not in {EditType.SUBSTITUTE, EditType.DISTORTED_SUBSTITUTE, EditType.ADD}:
                continue
            if not operation.target_phone_indices:
                continue

            target_index = operation.target_phone_indices[0]
            target_node = node_map.get(target_index)
            if target_node is None:
                continue

            start_sec, end_sec = helper._fragment_timing(target_node, operation, budget_map.get(target_index))
            phones = helper._phones_for_operation(target_node, operation)
            text = self._phones_to_text(phones, operation.edit_type)
            samples = self._render_with_coqui(text, audio.sample_rate_hz)
            source_name = self.synthesizer_name

            if not samples:
                fallback_fragment = fallback_say.synthesize(
                    audio,
                    graph,
                    EditPlan(operations=[operation], planner_name="coqui_fallback"),
                    timing,
                )
                if fallback_fragment:
                    samples = list(fallback_fragment[0].samples or [])
                    source_name = fallback_say.synthesizer_name

            if not samples:
                samples = helper._render_phone_sequence(
                    phones=phones,
                    sample_rate_hz=audio.sample_rate_hz,
                    duration_sec=max(0.03, end_sec - start_sec),
                    distorted=(operation.edit_type == EditType.DISTORTED_SUBSTITUTE),
                    seed=target_index,
                )
                source_name = helper.synthesizer_name

            target_samples = max(1, int(round(max(0.03, end_sec - start_sec) * audio.sample_rate_hz)))
            samples = self._resample(samples, target_samples)
            if operation.edit_type == EditType.DISTORTED_SUBSTITUTE:
                samples = self._distort(samples)

            fragments.append(
                AudioSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    samples=samples,
                    label="{0}:{1}".format(operation.edit_type.value, "-".join(phones)),
                    source=source_name,
                )
            )
        return fragments

    def _phones_to_text(self, phones: List[str], edit_type: EditType) -> str:
        mapping = {
            "AH": "uh",
            "AE": "a",
            "EH": "eh",
            "IH": "ih",
            "IY": "ee",
            "OW": "oh",
            "UW": "oo",
            "AY": "eye",
            "EY": "ay",
            "OY": "oy",
            "AW": "ow",
            "ER": "er",
            "R": "r",
            "L": "l",
            "W": "w",
            "Y": "y",
            "M": "m",
            "N": "n",
            "NG": "ng",
            "B": "b",
            "D": "d",
            "G": "g",
            "P": "p",
            "T": "t",
            "K": "k",
            "S": "s",
            "Z": "z",
            "SH": "sh",
            "TH": "th",
            "DH": "th",
            "F": "f",
            "V": "v",
            "CH": "ch",
            "JH": "j",
            "HH": "h",
        }
        pieces = [mapping.get(phone, phone.lower()) for phone in phones]
        if edit_type == EditType.ADD:
            return " ".join(pieces)
        return "".join(pieces) if len(pieces) > 1 else pieces[0]

    def _render_with_coqui(
        self,
        text: str,
        target_sample_rate_hz: int,
        prephonemized: bool = False,
        language: Optional[str] = None,
    ) -> List[float]:
        fd_script, script_path = tempfile.mkstemp(suffix=".py")
        os.close(fd_script)
        fd_wav, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_wav)
        try:
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(self._coqui_script())
            env = os.environ.copy()
            env["COQUI_TEXT"] = text
            env["COQUI_OUT"] = wav_path
            env["COQUI_MODEL"] = self.model_name
            env["COQUI_USE_GPU"] = "1" if self.use_gpu else "0"
            env["COQUI_SPEAKER_WAV"] = self.speaker_wav_path or ""
            env["COQUI_SPEAKER"] = self.speaker_name or ""
            env["COQUI_PREPHONEMIZED"] = "1" if prephonemized else "0"
            env["COQUI_LANGUAGE"] = language or ""
            completed = subprocess.run(
                resolve_python_command(self.conda_env_name) + [script_path],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                details = stderr or stdout or f"Coqui subprocess exited with code {completed.returncode}."
                raise CoquiSynthesisError(details)
            if not os.path.exists(wav_path):
                raise CoquiSynthesisError("Coqui did not write the expected WAV output.")
            samples, source_rate = self._read_wav(wav_path)
            if not samples:
                raise CoquiSynthesisError("Coqui wrote an empty or unreadable WAV output.")
            if source_rate != target_sample_rate_hz:
                target_length = int(round(len(samples) * float(target_sample_rate_hz) / float(source_rate)))
                samples = self._resample(samples, max(1, target_length))
            return self._trim_silence(samples)
        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def _coqui_script(self) -> str:
        return (
            "import os\n"
            "from TTS.api import TTS\n"
            "class IdentityPhonemizer:\n"
            "    def phonemize(self, text, separator='', language=None):\n"
            "        return text\n"
            "    def print_logs(self, level=0):\n"
            "        return None\n"
            "text = os.environ['COQUI_TEXT']\n"
            "out_path = os.environ['COQUI_OUT']\n"
            "model_name = os.environ['COQUI_MODEL']\n"
            "use_gpu = os.environ.get('COQUI_USE_GPU', '0') == '1'\n"
            "speaker_wav = os.environ.get('COQUI_SPEAKER_WAV') or None\n"
            "speaker_name = os.environ.get('COQUI_SPEAKER') or None\n"
            "prephonemized = os.environ.get('COQUI_PREPHONEMIZED', '0') == '1'\n"
            "language = os.environ.get('COQUI_LANGUAGE') or None\n"
            "tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)\n"
            "if prephonemized:\n"
            "    synthesizer = tts.synthesizer\n"
            "    if synthesizer is None or getattr(synthesizer, 'tts_model', None) is None:\n"
            "        raise RuntimeError('Coqui synthesizer did not initialize a TTS model.')\n"
            "    tokenizer = getattr(synthesizer.tts_model, 'tokenizer', None)\n"
            "    if tokenizer is None:\n"
            "        raise RuntimeError('Loaded Coqui model does not expose a tokenizer.')\n"
            "    if not getattr(tokenizer, 'use_phonemes', False):\n"
            "        raise RuntimeError('Loaded Coqui model is not configured for phoneme input.')\n"
            "    tokenizer.phonemizer = IdentityPhonemizer()\n"
            "    tokenizer.text_cleaner = None\n"
            "kwargs = {'text': text, 'file_path': out_path}\n"
            "if speaker_name:\n"
            "    kwargs['speaker'] = speaker_name\n"
            "if speaker_wav:\n"
            "    kwargs['speaker_wav'] = speaker_wav\n"
            "if language:\n"
            "    kwargs['language'] = language\n"
            "kwargs['split_sentences'] = False\n"
            "tts.tts_to_file(**kwargs)\n"
        )

    def _read_wav(self, path: str) -> Tuple[List[float], int]:
        with wave.open(path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width != 2:
            return [], frame_rate
        integers = [
            int.from_bytes(raw[index : index + 2], byteorder="little", signed=True)
            for index in range(0, len(raw), 2)
        ]
        if channel_count > 1:
            mono: List[float] = []
            for start in range(0, len(integers), channel_count):
                frame = integers[start : start + channel_count]
                mono.append(sum(frame) / float(len(frame)))
            integers = mono
        return ([float(value) / 32768.0 for value in integers], frame_rate)

    def _trim_silence(self, samples: List[float]) -> List[float]:
        if not samples:
            return []
        threshold = 0.01
        start = 0
        end = len(samples)
        while start < len(samples) and abs(samples[start]) < threshold:
            start += 1
        while end > start and abs(samples[end - 1]) < threshold:
            end -= 1
        trimmed = samples[start:end]
        return trimmed if trimmed else samples

    def _resample(self, samples: List[float], target_length: int) -> List[float]:
        if target_length <= 0:
            return []
        if not samples:
            return [0.0] * target_length
        if len(samples) == target_length:
            return list(samples)
        if len(samples) == 1:
            return [samples[0]] * target_length
        output: List[float] = []
        scale = float(len(samples) - 1) / float(max(target_length - 1, 1))
        for index in range(target_length):
            position = index * scale
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            mix = position - left
            output.append((samples[left] * (1.0 - mix)) + (samples[right] * mix))
        return output

    def _distort(self, samples: List[float]) -> List[float]:
        distorted: List[float] = []
        for index, sample in enumerate(samples):
            scaled = sample * (0.82 if (index // 9) % 2 == 0 else 0.58)
            distorted.append(max(-1.0, min(1.0, scaled)))
        return distorted


class CoquiSynthesisError(RuntimeError):
    """Raised when the Coqui subprocess fails with useful stderr."""
