from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .edits import EditOperation
from .timing import DurationBudget


@dataclass
class SliderControls:
    jibberish: float
    clarity: float
    timing_instability: float
    allow_full_gibberish: bool = False


@dataclass
class PronunciationSkeleton:
    word: str
    word_index: int
    phone_sequence: list[str]
    phone_indices: list[int]
    anchor_phone_indices: list[int]
    anchor_phones: list[str]
    primary_vowel_index: Optional[int]
    primary_vowel_phone: Optional[str]
    stress_bearing_phone_indices: list[int] = field(default_factory=list)
    syllable_count: int = 1
    onset_rime_shape: str = ""
    word_boundary_shape: str = ""
    manner_classes: list[str] = field(default_factory=list)
    is_short_word: bool = False


@dataclass
class ProtectionMap:
    protected_phone_indices: list[int]
    anchor_phone_indices: list[int]
    primary_vowel_index: Optional[int]
    preserve_phone_order: bool
    preserve_syllable_count: bool


@dataclass
class SimilarityComponents:
    anchor_phone_preservation: float
    vowel_nucleus_similarity: float
    syllable_count_similarity: float
    phone_order_similarity: float
    duration_shape_similarity: float


@dataclass
class PronunciationSimilarityScore:
    score: float
    threshold: float
    passed: bool
    components: SimilarityComponents


@dataclass
class RepairAction:
    name: str
    applied: bool
    details: str = ""


@dataclass
class ClarityPlan:
    parameters: dict[str, float] = field(default_factory=dict)
    guardrails: dict[str, object] = field(default_factory=dict)


@dataclass
class TimingInstabilityPlan:
    budgets: list[DurationBudget] = field(default_factory=list)
    pause_after_phone_indices: list[int] = field(default_factory=list)
    onset_repeat_phone_indices: list[int] = field(default_factory=list)
    micro_stutter_phone_indices: list[int] = field(default_factory=list)
    total_requested_delta_sec: float = 0.0
    total_compensated_delta_sec: float = 0.0
    max_total_duration_drift_ratio: float = 0.0
    preserve_phone_order: bool = True


@dataclass
class PronunciationSafePlan:
    controls: SliderControls
    skeletons: list[PronunciationSkeleton]
    protection_maps: list[ProtectionMap]
    operations: list[EditOperation]
    clarity_plan: ClarityPlan
    similarity: PronunciationSimilarityScore
    repairs: list[RepairAction] = field(default_factory=list)
    generated_parameters: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class AcceptanceCaseResult:
    name: str
    word: str
    expected: str
    passed: bool
    similarity_score: float
    similarity_threshold: float
    operation_count: int
    timing_budget_count: int
    preserve_phone_order: bool


@dataclass
class PronunciationAcceptanceReport:
    config_name: str
    case_results: list[AcceptanceCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.case_results)
