from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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


@dataclass
class RangeConfig:
    min: float
    max: float
    meaning: Optional[str] = None


@dataclass
class GlobalConstraintsConfig:
    slider_range: RangeConfig
    time_range: RangeConfig
    preserve_core_pronunciation: bool
    allow_full_gibberish: bool
    minimum_pronunciation_similarity: float
    minimum_pronunciation_similarity_short_word: float
    max_total_duration_drift_ratio: float
    random_seed: int


@dataclass
class AnchorSelectionRulesConfig:
    always_anchor_primary_vowel: bool
    always_anchor_first_content_consonant: bool
    anchor_final_consonant_for_closed_syllables: bool
    short_word_anchor_all_phones_when_phone_count_lte: int


@dataclass
class PronunciationSkeletonConfig:
    extract: list[str]
    anchor_selection_rules: AnchorSelectionRulesConfig


@dataclass
class PronunciationSimilarityWeightsConfig:
    anchor_phone_preservation: float
    vowel_nucleus_similarity: float
    syllable_count_similarity: float
    phone_order_similarity: float
    duration_shape_similarity: float


@dataclass
class PronunciationSimilarityScoreConfig:
    range: list[float]
    weights: PronunciationSimilarityWeightsConfig
    repair_order: list[str]


@dataclass
class ShortWordModeConfig:
    enabled_when_phone_count_lte: int
    max_phone_substitution_ratio: float
    max_deletions_per_word: int
    preserve_first_phone: bool
    preserve_primary_vowel_nucleus: bool


@dataclass
class SliderGuardrailsConfig:
    values: dict[str, object] = field(default_factory=dict)
    short_word_mode: Optional[ShortWordModeConfig] = None


@dataclass
class SliderConfig:
    label: str
    default: float
    user_description: str
    independence_rule: str
    pronunciation_guardrails: SliderGuardrailsConfig
    generated_parameters: dict[str, object] = field(default_factory=dict)
    allowed_examples_for_no: list[str] = field(default_factory=list)
    disallowed_examples_for_no: list[str] = field(default_factory=list)


@dataclass
class ProcessingStageConfig:
    id: str
    outputs: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)
    condition: Optional[str] = None


@dataclass
class OverrideModeConfig:
    values: dict[str, object] = field(default_factory=dict)


@dataclass
class AcceptanceTestConfig:
    name: str
    word: str
    phones: list[str]
    controls: dict[str, float]
    expected: str


@dataclass
class PronunciationSliderConfig:
    version: int
    kind: str
    name: str
    description: str
    global_constraints: GlobalConstraintsConfig
    pronunciation_skeleton: PronunciationSkeletonConfig
    pronunciation_similarity_score: PronunciationSimilarityScoreConfig
    sliders: dict[str, SliderConfig]
    processing_pipeline: list[ProcessingStageConfig] = field(default_factory=list)
    override_modes: dict[str, OverrideModeConfig] = field(default_factory=dict)
    acceptance_tests: list[AcceptanceTestConfig] = field(default_factory=list)
