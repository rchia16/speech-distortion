from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeverityConfig:
    global_severity: float
    complexity_slope: float
    distortion_bias: float
    substitution_bias: float
    addition_bias: float
    vowel_lengthening_bias: float
    consonant_lengthening_bias: float


@dataclass
class AlignmentConfig:
    runtime_backend: str
    offline_backend: str
    emit_phone_confidence: bool


@dataclass
class PhonologyConfig:
    g2p_backend: str
    language: str
    compute_complexity: bool


@dataclass
class EditingConfig:
    source_editor_backend: str
    enable_vowel_lengthening: bool
    enable_consonant_lengthening: bool
    enable_distortion_shaping: bool


@dataclass
class ResynthesisConfig:
    backend: str
    use_context_phones: int
    max_fragment_ms: int


@dataclass
class TimbreConfig:
    backend: str
    enabled: bool


@dataclass
class StitchingConfig:
    crossfade_ms: int
    smooth_energy: bool
    smooth_f0: bool


@dataclass
class PipelineConfig:
    sample_rate_hz: int
    device: str
    preserve_total_duration: bool
    target_latency_ms: int
    severity: SeverityConfig
    alignment: AlignmentConfig
    phonology: PhonologyConfig
    editing: EditingConfig
    resynthesis: ResynthesisConfig
    timbre: TimbreConfig
    stitching: StitchingConfig
