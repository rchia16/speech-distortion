from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.io import wavfile
from speechbrain.augment.time_domain import (
    AddNoise,
    AddReverb,
    ChannelDrop,
    ChannelSwap,
    DoClip,
    DropFreq,
    DropBitResolution,
    RandAmp,
    SignFlip,
    SpeedPerturb,
)


ROOT = Path(__file__).resolve().parent
INPUT_WAV = ROOT / "yes_slow.wav"
SEGMENTED_INPUTS = [
    # ROOT / "thetimehascome.wav",
    ROOT / "yes_slow.wav",
]
AUG_DIR = ROOT / "speechbrain_aug_assets"
CORRUPTION_AMOUNT = 1.0
CORRUPTION_RAMP_MS = 10
CORRUPTION_SEGMENTS = [
    {"start": 0.2, "end": 0.4, "amount": 1.0},
    {"start": 0.4, "end": 0.6, "amount": 0.6},
]


LEVELS = {
    "minimal": {
        "amp_low": 0.92,
        "amp_high": 1.05,
        "snr_low": 24,
        "snr_high": 30,
        "rir_scale_factor": 0.9,
        "drop_freq_count_low": 1,
        "drop_freq_count_high": 1,
        "drop_freq_width": 0.03,
        "seed": 11,
    },
    "medium": {
        "amp_low": 0.72,
        "amp_high": 1.18,
        "snr_low": 13,
        "snr_high": 19,
        "rir_scale_factor": 1.0,
        "drop_freq_count_low": 1,
        "drop_freq_count_high": 2,
        "drop_freq_width": 0.06,
        "seed": 23,
    },
    "high": {
        "amp_low": 0.52,
        "amp_high": 1.30,
        "snr_low": -1,
        "snr_high": 4,
        "rir_scale_factor": 1.2,
        "drop_freq_count_low": 2,
        "drop_freq_count_high": 4,
        "drop_freq_width": 0.09,
        "seed": 37,
    },
    "radio": {
        "amp_low": 0.80,
        "amp_high": 1.10,
        "snr_low": 8,
        "snr_high": 14,
        "rir_scale_factor": 0.95,
        "drop_freq_count_low": 2,
        "drop_freq_count_high": 3,
        "drop_freq_width": 0.07,
        "clip_low": 0.22,
        "clip_high": 0.35,
        "bit_depth": "int8",
        "seed": 41,
    },
    "lofi": {
        "amp_low": 0.70,
        "amp_high": 1.08,
        "snr_low": 10,
        "snr_high": 16,
        "rir_scale_factor": 1.05,
        "drop_freq_count_low": 1,
        "drop_freq_count_high": 2,
        "drop_freq_width": 0.08,
        "bit_depth": "int8",
        "speed": [96],
        "seed": 53,
    },
    "corrupted": {
        "amp_low": 0.63,
        "amp_high": 1.28,
        "snr_low": 0,
        "snr_high": 4,
        "rir_scale_factor": 1.18,
        "drop_freq_count_low": 3,
        "drop_freq_count_high": 4,
        "drop_freq_width": 0.08,
        "clip_low": 0.19,
        "clip_high": 0.31,
        "bit_depth": "int8",
        "speed": [90],
        "sign_flip_prob": 1.0,
        "channel_drop_rate": 0.08,
        "channel_swap_min": 1,
        "channel_swap_max": 1,
        "seed": 67,
    },
}


class LocalAddReverb(AddReverb):
    """Use SpeechBrain's AddReverb with a direct local RIR file."""

    def __init__(self, rir_path: Path, **kwargs):
        super().__init__(csv_file=str(rir_path), **kwargs)
        self.rir_path = Path(rir_path)

    def _load_rir(self, waveforms: torch.Tensor) -> torch.Tensor:
        rir_np, rir_sample_rate = sf.read(
            str(self.rir_path), dtype="float32", always_2d=True
        )
        rir_waveform = torch.from_numpy(rir_np).unsqueeze(0).to(waveforms.dtype)
        if self.reverb_sample_rate != rir_sample_rate:
            raise ValueError(
                f"RIR sample rate {rir_sample_rate} does not match expected "
                f"{self.reverb_sample_rate}"
            )
        return rir_waveform.to(waveforms.device)


def create_rir(sample_rate: int) -> Path:
    """Create a short synthetic room impulse response for AddReverb."""
    AUG_DIR.mkdir(exist_ok=True)
    rir_path = AUG_DIR / "synthetic_rir.wav"

    duration_s = 0.9
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    rir = np.zeros_like(t, dtype=np.float32)
    rir[0] = 1.0

    early_reflections = [
        (0.012, 0.45),
        (0.028, 0.30),
        (0.051, 0.22),
        (0.083, 0.16),
    ]
    for delay_s, gain in early_reflections:
        idx = min(int(delay_s * sample_rate), len(rir) - 1)
        rir[idx] += gain

    decay = np.exp(-4.8 * t).astype(np.float32)
    late_tail = np.random.default_rng(7).normal(0.0, 0.035, size=len(t)).astype(np.float32)
    rir += decay * late_tail
    rir /= np.max(np.abs(rir)) + 1e-8

    sf.write(rir_path, rir, sample_rate)
    return rir_path


def clamp_audio(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0.999:
        waveform = waveform / peak * 0.999
    return waveform


def ensure_3d(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 2:
        return waveform.unsqueeze(-1)
    return waveform


def normalize_waveform_shape(waveform: torch.Tensor) -> torch.Tensor:
    waveform = ensure_3d(waveform)
    if waveform.ndim == 3 and waveform.shape[1] < waveform.shape[2]:
        waveform = waveform.transpose(1, 2)
    return waveform


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)


def linear_scale(amount: float) -> float:
    return max(0.0, min(1.0, amount))


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def scale_corrupted_cfg(amount: float) -> dict[str, float | int]:
    scaled = linear_scale(amount)
    target = LEVELS["corrupted"]

    cfg: dict[str, float | int] = {
        "amp_low": lerp(1.0, float(target["amp_low"]), scaled),
        "amp_high": lerp(1.0, float(target["amp_high"]), scaled),
        "snr_low": lerp(60.0, float(target["snr_low"]), scaled),
        "snr_high": lerp(60.0, float(target["snr_high"]), scaled),
        "rir_scale_factor": lerp(1.0, float(target["rir_scale_factor"]), scaled),
        "drop_freq_count_low": int(round(lerp(0.0, float(target["drop_freq_count_low"]), scaled))),
        "drop_freq_count_high": int(round(lerp(0.0, float(target["drop_freq_count_high"]), scaled))),
        "drop_freq_width": lerp(0.0, float(target["drop_freq_width"]), scaled),
        "clip_low": lerp(1.0, float(target["clip_low"]), scaled),
        "clip_high": lerp(1.0, float(target["clip_high"]), scaled),
        "channel_drop_rate": lerp(0.0, float(target["channel_drop_rate"]), scaled),
        "seed": int(target["seed"]),
    }

    if scaled > 0.12:
        cfg["bit_depth"] = str(target["bit_depth"])
    if scaled > 0.18:
        speed_target = float(target["speed"][0])
        cfg["speed"] = [int(round(lerp(100.0, speed_target, scaled)))]
    if scaled > 0.3:
        cfg["sign_flip_prob"] = lerp(0.0, float(target["sign_flip_prob"]), scaled)
    if scaled > 0.45:
        cfg["channel_swap_min"] = int(target["channel_swap_min"])
        cfg["channel_swap_max"] = int(target["channel_swap_max"])

    cfg["drop_freq_count_high"] = max(
        int(cfg["drop_freq_count_low"]), int(cfg["drop_freq_count_high"])
    )
    return cfg


def match_length(waveform: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = waveform.shape[1]
    if current_frames == target_frames:
        return waveform
    if current_frames > target_frames:
        return waveform[:, :target_frames, :]

    pad = waveform[:, -1:, :].expand(-1, target_frames - current_frames, -1)
    return torch.cat([waveform, pad], dim=1)


def render_corrupted_waveform(
    waveform: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
    rir_path: Path,
    cfg: dict[str, float | int],
) -> torch.Tensor:
    torch.manual_seed(int(cfg["seed"]))

    augmented = normalize_waveform_shape(waveform.clone())
    augmented = RandAmp(
        amp_low=float(cfg["amp_low"]),
        amp_high=float(cfg["amp_high"]),
    )(augmented)
    augmented = AddNoise(
        snr_low=float(cfg["snr_low"]),
        snr_high=float(cfg["snr_high"]),
        normalize=True,
        clean_sample_rate=sample_rate,
    )(augmented, lengths)
    augmented = LocalAddReverb(
        rir_path=rir_path,
        rir_scale_factor=float(cfg["rir_scale_factor"]),
        reverb_sample_rate=sample_rate,
        clean_sample_rate=sample_rate,
    )(augmented)
    if "speed" in cfg:
        augmented = SpeedPerturb(
            orig_freq=sample_rate,
            speeds=list(cfg["speed"]),
        )(augmented)
        augmented = match_length(augmented, waveform.shape[1])
    if int(cfg["drop_freq_count_high"]) > 0 and float(cfg["drop_freq_width"]) > 0:
        augmented = DropFreq(
            drop_freq_count_low=int(cfg["drop_freq_count_low"]),
            drop_freq_count_high=int(cfg["drop_freq_count_high"]),
            drop_freq_width=float(cfg["drop_freq_width"]),
        )(augmented)
    if "clip_low" in cfg:
        augmented = DoClip(
            clip_low=float(cfg["clip_low"]),
            clip_high=float(cfg["clip_high"]),
        )(augmented)
    if "bit_depth" in cfg:
        augmented = DropBitResolution(target_dtype=str(cfg["bit_depth"]))(augmented)
    if "sign_flip_prob" in cfg:
        augmented = SignFlip(flip_prob=float(cfg["sign_flip_prob"]))(augmented)
    if "channel_drop_rate" in cfg:
        augmented = ChannelDrop(drop_rate=float(cfg["channel_drop_rate"]))(augmented)
    if "channel_swap_min" in cfg:
        augmented = ChannelSwap(
            min_swap=int(cfg["channel_swap_min"]),
            max_swap=int(cfg["channel_swap_max"]),
        )(augmented)
    return normalize_waveform_shape(clamp_audio(augmented))


def build_corruption_envelope(
    total_frames: int,
    sample_rate: int,
    segments: list[dict[str, float | int]],
    ramp_ms: int,
) -> torch.Tensor:
    envelope = torch.zeros(total_frames, dtype=torch.float32)
    ramp_frames = max(1, int(sample_rate * ramp_ms / 1000))

    for segment in segments:
        start_ratio = max(0.0, min(1.0, float(segment["start"])))
        end_ratio = max(0.0, min(1.0, float(segment["end"])))
        start = max(0, int(total_frames * start_ratio))
        end = min(total_frames, int(total_frames * end_ratio))
        amount = max(0.0, min(1.0, float(segment["amount"])))
        if end <= start or amount <= 0.0:
            continue

        envelope[start:end] = torch.maximum(envelope[start:end], torch.full((end - start,), amount))

    nonzero = envelope > 0
    idx = 0
    while idx < total_frames:
        if not nonzero[idx]:
            idx += 1
            continue

        block_start = idx
        while idx < total_frames and nonzero[idx]:
            idx += 1
        block_end = idx

        block_length = block_end - block_start
        local_ramp = min(ramp_frames, max(1, block_length // 2))
        if local_ramp <= 0:
            continue

        start_target = envelope[block_start]
        end_target = envelope[block_end - 1]

        ramp_up = torch.linspace(0.0, float(start_target), local_ramp)
        envelope[block_start : block_start + local_ramp] = torch.minimum(
            envelope[block_start : block_start + local_ramp], ramp_up
        )

        ramp_down = torch.linspace(float(end_target), 0.0, local_ramp)
        envelope[block_end - local_ramp : block_end] = torch.minimum(
            envelope[block_end - local_ramp : block_end], ramp_down
        )

    return envelope.view(1, total_frames, 1)


def apply_segmented_corruption(
    waveform: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
    rir_path: Path,
    segments: list[dict[str, float | int]],
    ramp_ms: int,
    output_stem: str,
) -> Path:
    max_amount = max((float(segment["amount"]) for segment in segments), default=0.0)
    output_path = ROOT / f"{output_stem}_segmented_corruption_distorted.wav"

    if max_amount <= 0.0:
        write_wav(output_path, waveform.squeeze(0).cpu().numpy(), sample_rate)
        return output_path

    corrupted_cfg = scale_corrupted_cfg(max_amount)
    corrupted_waveform = render_corrupted_waveform(
        waveform, lengths, sample_rate, rir_path, corrupted_cfg
    )
    envelope = build_corruption_envelope(
        waveform.shape[1], sample_rate, segments, ramp_ms
    ).to(waveform.device)
    clean_waveform = normalize_waveform_shape(waveform)
    blended = clamp_audio(
        clean_waveform * (1.0 - envelope) + corrupted_waveform * envelope
    )

    write_wav(output_path, blended.squeeze(0).cpu().numpy(), sample_rate)
    return output_path


def apply_level(
    waveform: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
    rir_path: Path,
    output_stem: str,
    level_name: str,
    cfg: dict[str, float | int],
) -> Path:
    if level_name == "corrupted_scaled":
        amount = float(cfg["corruption_amount"])
        output_path = ROOT / f"{output_stem}_corrupted_scaled_distorted.wav"
        if amount <= 0.0:
            write_wav(output_path, waveform.squeeze(0).cpu().numpy(), sample_rate)
            return output_path
        cfg = scale_corrupted_cfg(amount)

    augmented = render_corrupted_waveform(waveform, lengths, sample_rate, rir_path, cfg)

    output = augmented.squeeze(0).cpu().numpy()
    output_path = ROOT / f"{output_stem}_{level_name}_distorted.wav"
    write_wav(output_path, output, sample_rate)
    return output_path


def main() -> None:
    waveform_np, sample_rate = sf.read(str(INPUT_WAV), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    lengths = torch.tensor([1.0], dtype=torch.float32)

    rir_path = create_rir(sample_rate)

    outputs = []
    for level_name, cfg in LEVELS.items():
        outputs.append(
            apply_level(
                waveform,
                lengths,
                sample_rate,
                rir_path,
                INPUT_WAV.stem,
                level_name,
                cfg,
            )
        )
    outputs.append(
        apply_level(
            waveform,
            lengths,
            sample_rate,
            rir_path,
            INPUT_WAV.stem,
            "corrupted_scaled",
            {"corruption_amount": CORRUPTION_AMOUNT, "seed": LEVELS["corrupted"]["seed"]},
        )
    )
    outputs.append(
        apply_segmented_corruption(
            waveform,
            lengths,
            sample_rate,
            rir_path,
            CORRUPTION_SEGMENTS,
            CORRUPTION_RAMP_MS,
            INPUT_WAV.stem,
        )
    )

    for input_path in SEGMENTED_INPUTS:
        if input_path == INPUT_WAV:
            continue
        waveform_np, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        lengths = torch.tensor([1.0], dtype=torch.float32)
        outputs.append(
            apply_segmented_corruption(
                waveform,
                lengths,
                sample_rate,
                rir_path,
                CORRUPTION_SEGMENTS,
                CORRUPTION_RAMP_MS,
                input_path.stem,
            )
        )

    for path in outputs:
        print(path.name)


if __name__ == "__main__":
    main()
