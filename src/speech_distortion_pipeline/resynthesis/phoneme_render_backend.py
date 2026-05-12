from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from speech_distortion_pipeline.models import AudioBuffer
from speech_distortion_pipeline.phonology.blabber_config import (
    COQUI_ENV_NAME,
    COQUI_WOMAN_MODEL_NAME,
    PHONE_TO_CLONE_TEXT,
    PHONE_TO_IPA,
)
from speech_distortion_pipeline.resynthesis.fragment_synthesizer import (
    CoquiFragmentSynthesizer,
    CoquiSynthesisError,
    resolve_python_command,
)


PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH = "coqui_tacotron2_ddc_ph"
PHONEME_RENDER_BACKEND_FASTPITCH = "fastpitch"
PHONEME_RENDER_BACKEND_KOKORO = "kokoro"
PHONEME_RENDER_BACKEND_PHONEME_VITS = "phoneme_vits"
PHONEME_RENDER_BACKENDS: tuple[str, str, str, str] = (
    PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    PHONEME_RENDER_BACKEND_FASTPITCH,
    PHONEME_RENDER_BACKEND_KOKORO,
    PHONEME_RENDER_BACKEND_PHONEME_VITS,
)

DEFAULT_FASTPITCH_RENDERER = os.environ.get("BLABBER_FASTPITCH_RENDERER", "").strip()
DEFAULT_FASTPITCH_MODEL = os.environ.get("BLABBER_FASTPITCH_MODEL", "nvidia/tts_en_fastpitch").strip()
DEFAULT_FASTPITCH_HIFIGAN_MODEL = os.environ.get("BLABBER_FASTPITCH_HIFIGAN_MODEL", "nvidia/tts_hifigan").strip()
DEFAULT_KOKORO_MODEL = os.environ.get("BLABBER_KOKORO_MODEL", "hexgrad/Kokoro-82M").strip()
KOKORO_VOICE_AF_HEART = "af_heart"
KOKORO_VOICE_AF_ALLOY = "af_alloy"
KOKORO_VOICE_AF_AOEDE = "af_aoede"
KOKORO_VOICE_AF_BELLA = "af_bella"
KOKORO_VOICE_AF_JESSICA = "af_jessica"
KOKORO_VOICE_AF_KORE = "af_kore"
KOKORO_VOICE_AF_NICOLE = "af_nicole"
KOKORO_VOICE_AF_NOVA = "af_nova"
KOKORO_VOICE_AF_RIVER = "af_river"
KOKORO_VOICE_AF_SARAH = "af_sarah"
KOKORO_VOICE_AF_SKY = "af_sky"
KOKORO_VOICE_AM_ADAM = "am_adam"
KOKORO_VOICE_AM_ECHO = "am_echo"
KOKORO_VOICE_AM_ERIC = "am_eric"
KOKORO_VOICE_AM_FENRIR = "am_fenrir"
KOKORO_VOICE_AM_LIAM = "am_liam"
KOKORO_VOICE_AM_MICHAEL = "am_michael"
KOKORO_VOICE_AM_ONYX = "am_onyx"
KOKORO_VOICE_AM_PUCK = "am_puck"
KOKORO_VOICE_AM_SANTA = "am_santa"
KOKORO_VOICE_BF_ALICE = "bf_alice"
KOKORO_VOICE_BF_EMMA = "bf_emma"
KOKORO_VOICE_BF_ISABELLA = "bf_isabella"
KOKORO_VOICE_BF_LILY = "bf_lily"
KOKORO_VOICE_BM_DANIEL = "bm_daniel"
KOKORO_VOICE_BM_FABLE = "bm_fable"
KOKORO_VOICE_BM_GEORGE = "bm_george"
KOKORO_VOICE_BM_LEWIS = "bm_lewis"
KOKORO_VOICE_GM_FABLE = "gm_fable"
KOKORO_VOICE_JF_ALPHA = "jf_alpha"
KOKORO_VOICE_JF_GONGITSUNE = "jf_gongitsune"
KOKORO_VOICE_JF_NEZUMI = "jf_nezumi"
KOKORO_VOICE_JF_TEBUKURO = "jf_tebukuro"
KOKORO_VOICE_JM_KUMO = "jm_kumo"
KOKORO_VOICE_ZF_XIAOBEI = "zf_xiaobei"
KOKORO_VOICE_ZF_XIAONI = "zf_xiaoni"
KOKORO_VOICE_ZF_XIAOXIAO = "zf_xiaoxiao"
KOKORO_VOICE_ZF_XIAOYI = "zf_xiaoyi"
KOKORO_VOICE_ZM_YUNJIAN = "zm_yunjian"
KOKORO_VOICE_ZM_YUNXI = "zm_yunxi"
KOKORO_VOICE_ZM_YUNXIA = "zm_yunxia"
KOKORO_VOICE_ZM_YUNYANG = "zm_yunyang"
KOKORO_VOICE_EF_DORA = "ef_dora"
KOKORO_VOICE_EM_ALEX = "em_alex"
KOKORO_VOICE_EM_SANTA = "em_santa"
KOKORO_VOICE_FF_SIWIS = "ff_siwis"
KOKORO_VOICE_HF_ALPHA = "hf_alpha"
KOKORO_VOICE_HF_BETA = "hf_beta"
KOKORO_VOICE_HM_OMEGA = "hm_omega"
KOKORO_VOICE_HM_PSI = "hm_psi"
KOKORO_VOICE_IF_SARA = "if_sara"
KOKORO_VOICE_IM_NICOLA = "im_nicola"
KOKORO_VOICE_PF_DORA = "pf_dora"
KOKORO_VOICE_PM_ALEX = "pm_alex"
KOKORO_VOICE_PM_SANTA = "pm_santa"
DEFAULT_KOKORO_VOICE = os.environ.get("BLABBER_KOKORO_VOICE", KOKORO_VOICE_BF_EMMA).strip().lower()
DEFAULT_KOKORO_MALE_VOICE = os.environ.get("BLABBER_KOKORO_MALE_VOICE", KOKORO_VOICE_BM_LEWIS).strip().lower()
KOKORO_VOICE_OPTIONS: tuple[str, ...] = (
    KOKORO_VOICE_AF_HEART,
    KOKORO_VOICE_AF_ALLOY,
    KOKORO_VOICE_AF_AOEDE,
    KOKORO_VOICE_AF_BELLA,
    KOKORO_VOICE_AF_JESSICA,
    KOKORO_VOICE_AF_KORE,
    KOKORO_VOICE_AF_NICOLE,
    KOKORO_VOICE_AF_NOVA,
    KOKORO_VOICE_AF_RIVER,
    KOKORO_VOICE_AF_SARAH,
    KOKORO_VOICE_AF_SKY,
    KOKORO_VOICE_AM_ADAM,
    KOKORO_VOICE_AM_ECHO,
    KOKORO_VOICE_AM_ERIC,
    KOKORO_VOICE_AM_FENRIR,
    KOKORO_VOICE_AM_LIAM,
    KOKORO_VOICE_AM_MICHAEL,
    KOKORO_VOICE_AM_ONYX,
    KOKORO_VOICE_AM_PUCK,
    KOKORO_VOICE_AM_SANTA,
    KOKORO_VOICE_BF_ALICE,
    KOKORO_VOICE_BF_EMMA,
    KOKORO_VOICE_BF_ISABELLA,
    KOKORO_VOICE_BF_LILY,
    KOKORO_VOICE_BM_DANIEL,
    KOKORO_VOICE_BM_FABLE,
    KOKORO_VOICE_BM_GEORGE,
    KOKORO_VOICE_BM_LEWIS,
    KOKORO_VOICE_GM_FABLE,
    KOKORO_VOICE_JF_ALPHA,
    KOKORO_VOICE_JF_GONGITSUNE,
    KOKORO_VOICE_JF_NEZUMI,
    KOKORO_VOICE_JF_TEBUKURO,
    KOKORO_VOICE_JM_KUMO,
    KOKORO_VOICE_ZF_XIAOBEI,
    KOKORO_VOICE_ZF_XIAONI,
    KOKORO_VOICE_ZF_XIAOXIAO,
    KOKORO_VOICE_ZF_XIAOYI,
    KOKORO_VOICE_ZM_YUNJIAN,
    KOKORO_VOICE_ZM_YUNXI,
    KOKORO_VOICE_ZM_YUNXIA,
    KOKORO_VOICE_ZM_YUNYANG,
    KOKORO_VOICE_EF_DORA,
    KOKORO_VOICE_EM_ALEX,
    KOKORO_VOICE_EM_SANTA,
    KOKORO_VOICE_FF_SIWIS,
    KOKORO_VOICE_HF_ALPHA,
    KOKORO_VOICE_HF_BETA,
    KOKORO_VOICE_HM_OMEGA,
    KOKORO_VOICE_HM_PSI,
    KOKORO_VOICE_IF_SARA,
    KOKORO_VOICE_IM_NICOLA,
    KOKORO_VOICE_PF_DORA,
    KOKORO_VOICE_PM_ALEX,
    KOKORO_VOICE_PM_SANTA,
)
KOKORO_VOICE_ALIASES: Mapping[str, str] = {
    **{voice_name: voice_name for voice_name in KOKORO_VOICE_OPTIONS if not voice_name.startswith("gm_")},
    KOKORO_VOICE_GM_FABLE: KOKORO_VOICE_BM_FABLE,
    "gm_lewis": KOKORO_VOICE_BM_LEWIS,
    "gm_daniel": KOKORO_VOICE_BM_DANIEL,
    "gm_george": KOKORO_VOICE_BM_GEORGE,
}
DEFAULT_PHONEME_VITS_MODEL = os.environ.get("BLABBER_PHONEME_VITS_MODEL", "").strip()


@dataclass(frozen=True)
class PhonemeRenderResult:
    samples: np.ndarray
    backend: str
    model_name: str
    input_encoding: str
    prephonemized: bool
    voice_name: str = ""


@dataclass(frozen=True)
class PhonemeRenderProbe:
    backend: str
    ok: bool
    status: str
    detail: str = ""
    model_name: str = ""


class PhonemeRenderError(RuntimeError):
    """Raised when a phoneme render backend fails."""


def normalize_phoneme_render_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in PHONEME_RENDER_BACKENDS:
        raise ValueError(
            f"Unsupported phoneme render backend '{value}'. Expected one of: {', '.join(PHONEME_RENDER_BACKENDS)}."
        )
    return normalized


def default_woman_phoneme_render_backend() -> str:
    return PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH


def probe_phoneme_render_backend(
    backend: str,
    conda_env_name: str = COQUI_ENV_NAME,
) -> PhonemeRenderProbe:
    normalized = normalize_phoneme_render_backend(backend)
    if normalized == PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH:
        return _probe_coqui_model(
            normalized,
            COQUI_WOMAN_MODEL_NAME,
            conda_env_name=conda_env_name,
        )
    if normalized == PHONEME_RENDER_BACKEND_PHONEME_VITS:
        if not DEFAULT_PHONEME_VITS_MODEL:
            return PhonemeRenderProbe(
                backend=normalized,
                ok=False,
                status="not_configured",
                detail="Set BLABBER_PHONEME_VITS_MODEL to a phoneme-trained Coqui VITS checkpoint.",
            )
        return _probe_coqui_model(
            normalized,
            DEFAULT_PHONEME_VITS_MODEL,
            conda_env_name=conda_env_name,
        )
    if normalized == PHONEME_RENDER_BACKEND_KOKORO:
        return _probe_kokoro_backend(normalized, conda_env_name=conda_env_name)
    if not DEFAULT_FASTPITCH_RENDERER:
        return _probe_fastpitch_backend(normalized)
    return _probe_fastpitch_backend(normalized)


def render_with_phoneme_backend(
    audio: AudioBuffer,
    phones: Sequence[str],
    backend: str = PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    conda_env_name: str = COQUI_ENV_NAME,
    backend_voice: str | None = None,
) -> PhonemeRenderResult:
    normalized = normalize_phoneme_render_backend(backend)
    if normalized == PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH:
        return _render_with_coqui_model(
            audio,
            phones,
            model_name=COQUI_WOMAN_MODEL_NAME,
            conda_env_name=conda_env_name,
            backend=normalized,
        )
    if normalized == PHONEME_RENDER_BACKEND_PHONEME_VITS:
        if not DEFAULT_PHONEME_VITS_MODEL:
            raise PhonemeRenderError(
                "BLABBER_PHONEME_VITS_MODEL is not set. Configure a phoneme-trained VITS model before using backend 'phoneme_vits'."
            )
        return _render_with_coqui_model(
            audio,
            phones,
            model_name=DEFAULT_PHONEME_VITS_MODEL,
            conda_env_name=conda_env_name,
            backend=normalized,
        )
    if normalized == PHONEME_RENDER_BACKEND_KOKORO:
        return _render_with_kokoro(
            audio,
            phones,
            backend=normalized,
            conda_env_name=conda_env_name,
            voice_name=backend_voice,
        )
    return _render_with_fastpitch_adapter(audio, phones, backend=normalized)


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


def normalize_kokoro_voice(value: str | None) -> tuple[str, str]:
    requested = (value or DEFAULT_KOKORO_VOICE or KOKORO_VOICE_BF_EMMA).strip().lower()
    canonical = KOKORO_VOICE_ALIASES.get(requested)
    if canonical is None:
        raise ValueError(
            f"Unsupported Kokoro voice '{value}'. Expected one of: {', '.join(KOKORO_VOICE_OPTIONS)}."
        )
    return requested, canonical


def phones_to_kokoro_markup(phones: Sequence[str]) -> str:
    surrogate_text = phones_to_clone_text(phones)
    ipa = phones_to_ipa(phones)
    return f"[{surrogate_text}](/{ipa}/)"


def _kokoro_lang_code_for_voice(voice_name: str) -> str:
    prefix = voice_name[:2].lower()
    lang_codes = {
        "af": "a",
        "am": "a",
        "bf": "b",
        "bm": "b",
        "jf": "j",
        "jm": "j",
        "zf": "z",
        "zm": "z",
        "ef": "e",
        "em": "e",
        "ff": "f",
        "hf": "h",
        "hm": "h",
        "if": "i",
        "im": "i",
        "pf": "p",
        "pm": "p",
    }
    if prefix in lang_codes:
        return lang_codes[prefix]
    raise ValueError(f"Unsupported Kokoro voice prefix for '{voice_name}'.")


def _probe_coqui_model(backend: str, model_name: str, conda_env_name: str) -> PhonemeRenderProbe:
    try:
        detail = _run_coqui_probe(model_name, conda_env_name)
    except PhonemeRenderError as exc:
        return PhonemeRenderProbe(
            backend=backend,
            ok=False,
            status="probe_failed",
            detail=str(exc),
            model_name=model_name,
        )
    return PhonemeRenderProbe(
        backend=backend,
        ok=True,
        status="phoneme_compatible",
        detail=detail,
        model_name=model_name,
    )


def _run_coqui_probe(model_name: str, conda_env_name: str) -> str:
    fd_script, script_path = tempfile.mkstemp(prefix="speech_distortion_probe_", suffix=".py")
    os.close(fd_script)
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(_coqui_probe_script())
        env = os.environ.copy()
        env["COQUI_MODEL"] = model_name
        completed = subprocess.run(
            resolve_python_command(conda_env_name) + [script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            raise PhonemeRenderError(detail or "Coqui probe failed.")
        return (completed.stdout or "").strip() or "phoneme-compatible"
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def _coqui_probe_script() -> str:
    return (
        "import os\n"
        "from TTS.api import TTS\n"
        "model_name = os.environ['COQUI_MODEL']\n"
        "tts = TTS(model_name=model_name, progress_bar=False, gpu=False)\n"
        "synthesizer = tts.synthesizer\n"
        "if synthesizer is None or getattr(synthesizer, 'tts_model', None) is None:\n"
        "    raise RuntimeError('Coqui synthesizer did not initialize a TTS model.')\n"
        "tokenizer = getattr(synthesizer.tts_model, 'tokenizer', None)\n"
        "if tokenizer is None:\n"
        "    raise RuntimeError('Loaded Coqui model does not expose a tokenizer.')\n"
        "if not getattr(tokenizer, 'use_phonemes', False):\n"
        "    raise RuntimeError('Loaded Coqui model is not configured for phoneme input.')\n"
        "print('phoneme-compatible')\n"
    )


def _render_with_coqui_model(
    audio: AudioBuffer,
    phones: Sequence[str],
    model_name: str,
    conda_env_name: str,
    backend: str,
) -> PhonemeRenderResult:
    synthesizer = CoquiFragmentSynthesizer(
        conda_env_name=conda_env_name,
        model_name=model_name,
    )
    try:
        rendered = synthesizer._render_with_coqui(
            phones_to_ipa(phones),
            audio.sample_rate_hz,
            prephonemized=True,
            language=None,
        )
    except CoquiSynthesisError as exc:
        raise PhonemeRenderError(str(exc)) from exc
    if not rendered:
        raise PhonemeRenderError(f"Backend '{backend}' produced no audio.")
    return PhonemeRenderResult(
        samples=np.asarray(rendered, dtype=np.float32),
        backend=backend,
        model_name=model_name,
        input_encoding="ipa",
        prephonemized=True,
    )


def _render_with_fastpitch_adapter(
    audio: AudioBuffer,
    phones: Sequence[str],
    backend: str,
) -> PhonemeRenderResult:
    if not DEFAULT_FASTPITCH_RENDERER:
        return _render_with_nemo_fastpitch(audio, phones, backend=backend)
    return _render_with_external_fastpitch_adapter(audio, phones, backend=backend)


def _probe_fastpitch_backend(backend: str) -> PhonemeRenderProbe:
    if DEFAULT_FASTPITCH_RENDERER:
        return PhonemeRenderProbe(
            backend=backend,
            ok=True,
            status="external_configured",
            detail=f"External renderer configured at '{DEFAULT_FASTPITCH_RENDERER}'.",
            model_name=DEFAULT_FASTPITCH_RENDERER,
        )
    fd_script, script_path = tempfile.mkstemp(prefix="speech_distortion_fastpitch_probe_", suffix=".py")
    os.close(fd_script)
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(_nemo_fastpitch_probe_script())
        env = os.environ.copy()
        env["FASTPITCH_MODEL"] = DEFAULT_FASTPITCH_MODEL
        env["FASTPITCH_HIFIGAN_MODEL"] = DEFAULT_FASTPITCH_HIFIGAN_MODEL
        completed = subprocess.run(
            [sys.executable, script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            return PhonemeRenderProbe(
                backend=backend,
                ok=False,
                status="probe_failed",
                detail=detail or "NeMo FastPitch probe failed.",
                model_name=DEFAULT_FASTPITCH_MODEL,
            )
        return PhonemeRenderProbe(
            backend=backend,
            ok=True,
            status="nemo_ready",
            detail=(completed.stdout or "").strip() or "nemo-fastpitch-ready",
            model_name=f"{DEFAULT_FASTPITCH_MODEL}+{DEFAULT_FASTPITCH_HIFIGAN_MODEL}",
        )
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def _probe_kokoro_backend(backend: str, conda_env_name: str) -> PhonemeRenderProbe:
    requested_voice, canonical_voice = normalize_kokoro_voice(DEFAULT_KOKORO_VOICE)
    fd_script, script_path = tempfile.mkstemp(prefix="speech_distortion_kokoro_probe_", suffix=".py")
    os.close(fd_script)
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(_kokoro_probe_script())
        env = os.environ.copy()
        env["KOKORO_MODEL"] = DEFAULT_KOKORO_MODEL
        env["KOKORO_VOICE"] = canonical_voice
        env["KOKORO_LANG_CODE"] = _kokoro_lang_code_for_voice(canonical_voice)
        completed = subprocess.run(
            resolve_python_command(conda_env_name) + [script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            return PhonemeRenderProbe(
                backend=backend,
                ok=False,
                status="probe_failed",
                detail=detail or "Kokoro probe failed.",
                model_name=DEFAULT_KOKORO_MODEL,
            )
        return PhonemeRenderProbe(
            backend=backend,
            ok=True,
            status="kokoro_ready",
            detail=(completed.stdout or "").strip() or f"kokoro-ready:{requested_voice}",
            model_name=DEFAULT_KOKORO_MODEL,
        )
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def _kokoro_probe_script() -> str:
    return (
        "import os\n"
        "from kokoro import KPipeline\n"
        "pipeline = KPipeline(lang_code=os.environ['KOKORO_LANG_CODE'])\n"
        "generator = pipeline('[test](/tEst/)', voice=os.environ['KOKORO_VOICE'])\n"
        "chunk = next(generator, None)\n"
        "if chunk is None:\n"
        "    raise RuntimeError('Kokoro produced no audio during probe.')\n"
        "print('kokoro-ready')\n"
    )


def _render_with_kokoro(
    audio: AudioBuffer,
    phones: Sequence[str],
    backend: str,
    conda_env_name: str,
    voice_name: str | None,
) -> PhonemeRenderResult:
    requested_voice, canonical_voice = normalize_kokoro_voice(voice_name)
    fd_script, script_path = tempfile.mkstemp(prefix="speech_distortion_kokoro_", suffix=".py")
    os.close(fd_script)
    fd_wav, wav_path = tempfile.mkstemp(prefix="speech_distortion_kokoro_", suffix=".wav")
    os.close(fd_wav)
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(_kokoro_render_script())
        env = os.environ.copy()
        env["KOKORO_MODEL"] = DEFAULT_KOKORO_MODEL
        env["KOKORO_VOICE"] = canonical_voice
        env["KOKORO_LANG_CODE"] = _kokoro_lang_code_for_voice(canonical_voice)
        env["KOKORO_TEXT"] = phones_to_kokoro_markup(phones)
        env["KOKORO_OUT"] = wav_path
        completed = subprocess.run(
            resolve_python_command(conda_env_name) + [script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            raise PhonemeRenderError(detail or "Kokoro render failed.")
        if not Path(wav_path).is_file():
            raise PhonemeRenderError("Kokoro did not produce the expected WAV output.")
        samples, source_rate = _read_wav_file(wav_path)
        if not samples:
            raise PhonemeRenderError("Kokoro wrote an empty or unreadable WAV output.")
        output = np.asarray(samples, dtype=np.float32)
        if source_rate != audio.sample_rate_hz:
            target_length = int(round(len(output) * float(audio.sample_rate_hz) / float(source_rate)))
            output = _resample_array(output, max(1, target_length))
        return PhonemeRenderResult(
            samples=output,
            backend=backend,
            model_name=DEFAULT_KOKORO_MODEL,
            input_encoding="pronunciation_markup",
            prephonemized=False,
            voice_name=requested_voice,
        )
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def _kokoro_render_script() -> str:
    return (
        "import os\n"
        "import numpy as np\n"
        "import soundfile as sf\n"
        "from kokoro import KPipeline\n"
        "pipeline = KPipeline(lang_code=os.environ['KOKORO_LANG_CODE'])\n"
        "generator = pipeline(os.environ['KOKORO_TEXT'], voice=os.environ['KOKORO_VOICE'])\n"
        "chunks = []\n"
        "for _graphemes, _phonemes, audio in generator:\n"
        "    if audio is None:\n"
        "        continue\n"
        "    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)\n"
        "    if chunk.size:\n"
        "        chunks.append(chunk)\n"
        "if not chunks:\n"
        "    raise RuntimeError('Kokoro produced no audio chunks.')\n"
        "sf.write(os.environ['KOKORO_OUT'], np.concatenate(chunks), 24000)\n"
    )


def _nemo_fastpitch_probe_script() -> str:
    return (
        "import os\n"
        "from nemo.collections.tts.models import FastPitchModel, HifiGanModel\n"
        "fastpitch = FastPitchModel.from_pretrained(os.environ['FASTPITCH_MODEL'])\n"
        "hifigan = HifiGanModel.from_pretrained(model_name=os.environ['FASTPITCH_HIFIGAN_MODEL'])\n"
        "print('nemo-fastpitch-ready')\n"
    )


def _render_with_nemo_fastpitch(
    audio: AudioBuffer,
    phones: Sequence[str],
    backend: str,
) -> PhonemeRenderResult:
    fd_script, script_path = tempfile.mkstemp(prefix="speech_distortion_fastpitch_", suffix=".py")
    os.close(fd_script)
    fd_wav, wav_path = tempfile.mkstemp(prefix="speech_distortion_fastpitch_", suffix=".wav")
    os.close(fd_wav)
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(_nemo_fastpitch_render_script())
        env = os.environ.copy()
        env["FASTPITCH_MODEL"] = DEFAULT_FASTPITCH_MODEL
        env["FASTPITCH_HIFIGAN_MODEL"] = DEFAULT_FASTPITCH_HIFIGAN_MODEL
        env["FASTPITCH_TEXT"] = phones_to_clone_text(phones)
        env["FASTPITCH_OUT"] = wav_path
        completed = subprocess.run(
            [sys.executable, script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            raise PhonemeRenderError(detail or "NeMo FastPitch render failed.")
        if not Path(wav_path).is_file():
            raise PhonemeRenderError("NeMo FastPitch did not produce the expected WAV output.")
        samples, source_rate = _read_wav_file(wav_path)
        if not samples:
            raise PhonemeRenderError("NeMo FastPitch wrote an empty or unreadable WAV output.")
        output = np.asarray(samples, dtype=np.float32)
        if source_rate != audio.sample_rate_hz:
            target_length = int(round(len(output) * float(audio.sample_rate_hz) / float(source_rate)))
            output = _resample_array(output, max(1, target_length))
        return PhonemeRenderResult(
            samples=output,
            backend=backend,
            model_name=f"{DEFAULT_FASTPITCH_MODEL}+{DEFAULT_FASTPITCH_HIFIGAN_MODEL}",
            input_encoding="clone_text",
            prephonemized=False,
        )
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def _nemo_fastpitch_render_script() -> str:
    return (
        "import os\n"
        "import soundfile as sf\n"
        "from nemo.collections.tts.models import FastPitchModel, HifiGanModel\n"
        "text = os.environ['FASTPITCH_TEXT']\n"
        "out_path = os.environ['FASTPITCH_OUT']\n"
        "fastpitch = FastPitchModel.from_pretrained(os.environ['FASTPITCH_MODEL'])\n"
        "hifigan = HifiGanModel.from_pretrained(model_name=os.environ['FASTPITCH_HIFIGAN_MODEL'])\n"
        "parsed = fastpitch.parse(text)\n"
        "spectrogram = fastpitch.generate_spectrogram(tokens=parsed)\n"
        "audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)\n"
        "sf.write(out_path, audio.to('cpu').detach().numpy()[0], 22050)\n"
    )


def _render_with_external_fastpitch_adapter(
    audio: AudioBuffer,
    phones: Sequence[str],
    backend: str,
) -> PhonemeRenderResult:
    fd_json, payload_path = tempfile.mkstemp(prefix="speech_distortion_fastpitch_", suffix=".json")
    os.close(fd_json)
    fd_wav, wav_path = tempfile.mkstemp(prefix="speech_distortion_fastpitch_", suffix=".wav")
    os.close(fd_wav)
    try:
        payload = {
            "phones": list(phones),
            "ipa": phones_to_ipa(phones),
            "sample_rate_hz": int(audio.sample_rate_hz),
            "output_wav_path": wav_path,
        }
        with open(payload_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        completed = subprocess.run(
            [DEFAULT_FASTPITCH_RENDERER, payload_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            raise PhonemeRenderError(detail or "FastPitch adapter failed.")
        if not Path(wav_path).is_file():
            raise PhonemeRenderError("FastPitch adapter did not produce the expected WAV output.")
        samples, source_rate = _read_wav_file(wav_path)
        if not samples:
            raise PhonemeRenderError("FastPitch adapter wrote an empty or unreadable WAV output.")
        output = np.asarray(samples, dtype=np.float32)
        if source_rate != audio.sample_rate_hz:
            target_length = int(round(len(output) * float(audio.sample_rate_hz) / float(source_rate)))
            output = _resample_array(output, max(1, target_length))
        return PhonemeRenderResult(
            samples=output,
            backend=backend,
            model_name=DEFAULT_FASTPITCH_RENDERER,
            input_encoding="ipa",
            prephonemized=True,
        )
    finally:
        if os.path.exists(payload_path):
            os.unlink(payload_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def _read_wav_file(path: str) -> tuple[list[float], int]:
    import wave

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
        mono: list[float] = []
        for start in range(0, len(integers), channel_count):
            frame = integers[start : start + channel_count]
            mono.append(sum(frame) / float(len(frame)))
        integers = mono
    return [float(value) / 32768.0 for value in integers], frame_rate


def _resample_array(samples: np.ndarray, target_length: int) -> np.ndarray:
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


def backend_metadata(result: PhonemeRenderResult) -> dict[str, str]:
    metadata = {
        "phoneme_render_backend": result.backend,
        "phoneme_render_model_name": result.model_name,
        "phoneme_render_input_encoding": result.input_encoding,
        "phoneme_render_prephonemized": "1" if result.prephonemized else "0",
    }
    if result.voice_name:
        metadata["kokoro_voice"] = result.voice_name
    return metadata
