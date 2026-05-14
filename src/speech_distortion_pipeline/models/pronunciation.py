from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .edits import EditOperation
from .timing import DurationBudget


@dataclass
class SliderControls:
    jibberish: float
    clarity: float
    timing_instability: float
    allow_full_gibberish: bool = False
    distance_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class DistanceSignal:
    name: str
    value: float
    sources: list[str] = field(default_factory=list)
    raw_features: dict[str, Any] = field(default_factory=dict)
    aligned_region: Optional[str] = None


@dataclass
class SliderEstimate:
    value: float
    confidence: float
    evidence_sources: list[str] = field(default_factory=list)
    routing_rule: Optional[str] = None


@dataclass
class RoutingDecision:
    slider_name: str
    value: float
    confidence: float
    evidence_sources: list[str] = field(default_factory=list)
    routing_rule: Optional[str] = None


@dataclass
class DistanceEvidenceBundle:
    signals: dict[str, DistanceSignal] = field(default_factory=dict)
    routing: dict[str, RoutingDecision] = field(default_factory=dict)


@dataclass
class CandidateScore:
    word: str
    word_index: int
    grapheme: str
    phones: list[str] = field(default_factory=list)
    grapheme_distance: float = 0.0
    phoneme_distance: float = 0.0
    pronunciation_similarity: float = 0.0
    accepted: bool = False
    rejection_reason: str = ""
    traversal_source: str = ""
    cost_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class TriphoneState:
    left: str
    center: str
    right: str
    left_boundary: bool = False
    right_boundary: bool = False
    stress: Optional[str] = None
    syllable_role: str = ""
    word_position: int = 0


@dataclass
class TriphoneFeatureVector:
    center_features: dict[str, Any] = field(default_factory=dict)
    left_right_compatibility: float = 0.0
    sonority_transition: float = 0.0
    consonant_cluster_legality: float = 0.0
    vowel_harmony: float = 0.0
    syllable_role_score: float = 0.0
    phonotactic_likelihood: float = 0.0


@dataclass
class TriphoneEdge:
    source: TriphoneState
    target: TriphoneState
    edge_cost: float
    feature_similarity: float
    phonotactic_likelihood: float
    hubert_weight: float
    perceptual_inertia: float
    cost_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class TraversalPath:
    mode: str
    proposal_source: str = ""
    states: list[TriphoneState] = field(default_factory=list)
    edges: list[TriphoneEdge] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    grapheme: str = ""
    total_cost: float = 0.0
    naturalness_score: float = 0.0
    continuity_score: float = 0.0
    diversity_score: float = 0.0
    reference_neighbors: list[str] = field(default_factory=list)
    feature_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    target_word: str
    target_graphemes: str
    target_phones: list[str] = field(default_factory=list)
    selected_grapheme_variant: str = ""
    selected_phones: list[str] = field(default_factory=list)
    pronunciation_similarity: float = 0.0
    slider_values: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    distance_evidence: dict[str, float] = field(default_factory=dict)
    distance_features: dict[str, dict[str, Any]] = field(default_factory=dict)
    routing_rules: dict[str, str] = field(default_factory=dict)
    acoustic_backend: str = "proxy"
    acoustic_model_path: str = ""
    acoustic_model_loaded: bool = False
    acoustic_model_compatibility: str = "unavailable"
    traversal_mode: str = ""
    traversal_source: str = ""
    traversal_path: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)


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
    slider_estimates: dict[str, SliderEstimate] = field(default_factory=dict)
    distance_evidence: dict[str, DistanceSignal] = field(default_factory=dict)
    evidence_bundle: Optional[DistanceEvidenceBundle] = None
    candidates: list[CandidateScore] = field(default_factory=list)
    selected_candidates: list[CandidateScore] = field(default_factory=list)
    traversal_paths: list[TraversalPath] = field(default_factory=list)
    trace: Optional[DecisionTrace] = None


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
