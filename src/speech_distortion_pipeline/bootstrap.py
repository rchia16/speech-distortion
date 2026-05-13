from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from speech_distortion_pipeline.alignment import Aligner, build_aligner
from speech_distortion_pipeline.config import AlignmentConfig, HybridConfig, PhonologyConfig, PipelineConfig
from speech_distortion_pipeline.editing import (
    PronunciationSourceEditor,
    SourceSegmentEditor,
    build_pronunciation_source_editor,
    build_source_editor,
)
from speech_distortion_pipeline.io import WavAudioReader
from speech_distortion_pipeline.phonology import (
    ComplexityScorer,
    FeatureExtractor,
    GraphemeToPhoneme,
    build_complexity_scorer,
    build_feature_extractor,
    build_grapheme_to_phoneme,
)
from speech_distortion_pipeline.planning import (
    ErrorPlanner,
    HybridPlanner,
    build_error_planner,
    build_hybrid_planner,
    build_severity_profile_from_config,
)
from speech_distortion_pipeline.resynthesis import (
    FragmentSynthesizer,
    PronunciationFragmentSynthesizer,
    build_fragment_synthesizer,
    build_pronunciation_fragment_synthesizer,
)
from speech_distortion_pipeline.stitching import AudioAssembler, build_audio_assembler
from speech_distortion_pipeline.timbre import TimbreProjector, build_timbre_projector
from speech_distortion_pipeline.models import SeverityProfile, SliderControls
from speech_distortion_pipeline.models import AcceptanceCaseResult, PronunciationAcceptanceReport
from speech_distortion_pipeline.timing import (
    DurationBudgetManager,
    PronunciationTimingPlanner,
    build_duration_budget_manager,
    build_hybrid_timing_planner,
)
from speech_distortion_pipeline.orchestration import SpeechDistortionPipeline


@dataclass
class FirstSlice:
    config: PipelineConfig
    aligner: Aligner


def build_first_slice(config: PipelineConfig) -> FirstSlice:
    return FirstSlice(
        config=config,
        aligner=build_aligner(config.alignment),
    )


@dataclass
class PhonologySlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer


def build_phonology_slice(config: PipelineConfig) -> PhonologySlice:
    g2p = build_grapheme_to_phoneme(config.phonology)
    return PhonologySlice(
        config=config,
        aligner=build_aligner(config.alignment),
        g2p=g2p,
        feature_extractor=build_feature_extractor(config.phonology, g2p),
        complexity_scorer=build_complexity_scorer(config.phonology),
    )


@dataclass
class PlanningSlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    severity_profile: SeverityProfile
    planner: ErrorPlanner


def build_planning_slice(config: PipelineConfig) -> PlanningSlice:
    phonology = build_phonology_slice(config)
    return PlanningSlice(
        config=config,
        aligner=phonology.aligner,
        g2p=phonology.g2p,
        feature_extractor=phonology.feature_extractor,
        complexity_scorer=phonology.complexity_scorer,
        severity_profile=build_severity_profile_from_config(config.severity),
        planner=build_error_planner(),
    )


@dataclass
class TimingSlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    severity_profile: SeverityProfile
    planner: ErrorPlanner
    budget_manager: DurationBudgetManager


def build_timing_slice(config: PipelineConfig) -> TimingSlice:
    planning = build_planning_slice(config)
    return TimingSlice(
        config=config,
        aligner=planning.aligner,
        g2p=planning.g2p,
        feature_extractor=planning.feature_extractor,
        complexity_scorer=planning.complexity_scorer,
        severity_profile=planning.severity_profile,
        planner=planning.planner,
        budget_manager=build_duration_budget_manager(),
    )


@dataclass
class EditingSlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    severity_profile: SeverityProfile
    planner: ErrorPlanner
    budget_manager: DurationBudgetManager
    source_editor: SourceSegmentEditor


def build_editing_slice(config: PipelineConfig) -> EditingSlice:
    timing = build_timing_slice(config)
    return EditingSlice(
        config=config,
        aligner=timing.aligner,
        g2p=timing.g2p,
        feature_extractor=timing.feature_extractor,
        complexity_scorer=timing.complexity_scorer,
        severity_profile=timing.severity_profile,
        planner=timing.planner,
        budget_manager=timing.budget_manager,
        source_editor=build_source_editor(),
    )


@dataclass
class ResynthesisSlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    severity_profile: SeverityProfile
    planner: ErrorPlanner
    budget_manager: DurationBudgetManager
    fragment_synthesizer: FragmentSynthesizer


def build_resynthesis_slice(config: PipelineConfig) -> ResynthesisSlice:
    timing = build_timing_slice(config)
    return ResynthesisSlice(
        config=config,
        aligner=timing.aligner,
        g2p=timing.g2p,
        feature_extractor=timing.feature_extractor,
        complexity_scorer=timing.complexity_scorer,
        severity_profile=timing.severity_profile,
        planner=timing.planner,
        budget_manager=timing.budget_manager,
        fragment_synthesizer=build_fragment_synthesizer(),
    )


@dataclass
class StitchingSlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    severity_profile: SeverityProfile
    planner: ErrorPlanner
    budget_manager: DurationBudgetManager
    source_editor: SourceSegmentEditor
    fragment_synthesizer: FragmentSynthesizer
    timbre_projector: TimbreProjector
    assembler: AudioAssembler


def build_stitching_slice(config: PipelineConfig) -> StitchingSlice:
    editing = build_editing_slice(config)
    resynthesis = build_resynthesis_slice(config)
    return StitchingSlice(
        config=config,
        aligner=editing.aligner,
        g2p=editing.g2p,
        feature_extractor=editing.feature_extractor,
        complexity_scorer=editing.complexity_scorer,
        severity_profile=editing.severity_profile,
        planner=editing.planner,
        budget_manager=editing.budget_manager,
        source_editor=editing.source_editor,
        fragment_synthesizer=resynthesis.fragment_synthesizer,
        timbre_projector=build_timbre_projector(),
        assembler=build_audio_assembler(),
    )


@dataclass
class TimbreSlice:
    config: PipelineConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    severity_profile: SeverityProfile
    planner: ErrorPlanner
    budget_manager: DurationBudgetManager
    fragment_synthesizer: FragmentSynthesizer
    timbre_projector: TimbreProjector


def build_timbre_slice(config: PipelineConfig) -> TimbreSlice:
    resynthesis = build_resynthesis_slice(config)
    return TimbreSlice(
        config=config,
        aligner=resynthesis.aligner,
        g2p=resynthesis.g2p,
        feature_extractor=resynthesis.feature_extractor,
        complexity_scorer=resynthesis.complexity_scorer,
        severity_profile=resynthesis.severity_profile,
        planner=resynthesis.planner,
        budget_manager=resynthesis.budget_manager,
        fragment_synthesizer=resynthesis.fragment_synthesizer,
        timbre_projector=build_timbre_projector(),
    )


@dataclass
class HybridPlanningSlice:
    config: HybridConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    planner: HybridPlanner


def build_hybrid_planning_slice(config: HybridConfig) -> HybridPlanningSlice:
    alignment = AlignmentConfig(
        runtime_backend="torchaudio_ctc",
        offline_backend="mfa",
        emit_phone_confidence=False,
    )
    phonology = PhonologyConfig(
        g2p_backend="espeak_ng",
        language="en-us",
        compute_complexity=False,
    )
    g2p = build_grapheme_to_phoneme(phonology)
    return HybridPlanningSlice(
        config=config,
        aligner=build_aligner(alignment),
        g2p=g2p,
        feature_extractor=build_feature_extractor(phonology, g2p),
        planner=build_hybrid_planner(config, g2p),
    )


@dataclass
class HybridTimingSlice:
    config: HybridConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    planner: HybridPlanner
    timing_planner: PronunciationTimingPlanner


def build_hybrid_timing_slice(config: HybridConfig) -> HybridTimingSlice:
    planning = build_hybrid_planning_slice(config)
    return HybridTimingSlice(
        config=config,
        aligner=planning.aligner,
        g2p=planning.g2p,
        feature_extractor=planning.feature_extractor,
        planner=planning.planner,
        timing_planner=build_hybrid_timing_planner(config),
    )


@dataclass
class HybridEditingSlice:
    config: HybridConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    planner: HybridPlanner
    timing_planner: PronunciationTimingPlanner
    source_editor: PronunciationSourceEditor


def build_hybrid_editing_slice(config: HybridConfig) -> HybridEditingSlice:
    timing = build_hybrid_timing_slice(config)
    return HybridEditingSlice(
        config=config,
        aligner=timing.aligner,
        g2p=timing.g2p,
        feature_extractor=timing.feature_extractor,
        planner=timing.planner,
        timing_planner=timing.timing_planner,
        source_editor=build_pronunciation_source_editor(),
    )


@dataclass
class HybridResynthesisSlice:
    config: HybridConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    planner: HybridPlanner
    timing_planner: PronunciationTimingPlanner
    source_editor: PronunciationSourceEditor
    fragment_synthesizer: PronunciationFragmentSynthesizer


def build_hybrid_resynthesis_slice(config: HybridConfig) -> HybridResynthesisSlice:
    editing = build_hybrid_editing_slice(config)
    return HybridResynthesisSlice(
        config=config,
        aligner=editing.aligner,
        g2p=editing.g2p,
        feature_extractor=editing.feature_extractor,
        planner=editing.planner,
        timing_planner=editing.timing_planner,
        source_editor=editing.source_editor,
        fragment_synthesizer=build_pronunciation_fragment_synthesizer(),
    )


@dataclass
class HybridTimbreSlice:
    config: HybridConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    planner: HybridPlanner
    timing_planner: PronunciationTimingPlanner
    source_editor: PronunciationSourceEditor
    fragment_synthesizer: PronunciationFragmentSynthesizer
    timbre_projector: TimbreProjector


def build_hybrid_timbre_slice(config: HybridConfig) -> HybridTimbreSlice:
    resynthesis = build_hybrid_resynthesis_slice(config)
    return HybridTimbreSlice(
        config=config,
        aligner=resynthesis.aligner,
        g2p=resynthesis.g2p,
        feature_extractor=resynthesis.feature_extractor,
        planner=resynthesis.planner,
        timing_planner=resynthesis.timing_planner,
        source_editor=resynthesis.source_editor,
        fragment_synthesizer=resynthesis.fragment_synthesizer,
        timbre_projector=build_timbre_projector(),
    )


@dataclass
class HybridStitchingSlice:
    config: HybridConfig
    aligner: Aligner
    g2p: GraphemeToPhoneme
    feature_extractor: FeatureExtractor
    planner: HybridPlanner
    timing_planner: PronunciationTimingPlanner
    source_editor: PronunciationSourceEditor
    fragment_synthesizer: PronunciationFragmentSynthesizer
    timbre_projector: TimbreProjector
    assembler: AudioAssembler


def build_hybrid_stitching_slice(config: HybridConfig) -> HybridStitchingSlice:
    timbre = build_hybrid_timbre_slice(config)
    return HybridStitchingSlice(
        config=timbre.config,
        aligner=timbre.aligner,
        g2p=timbre.g2p,
        feature_extractor=timbre.feature_extractor,
        planner=timbre.planner,
        timing_planner=timbre.timing_planner,
        source_editor=timbre.source_editor,
        fragment_synthesizer=timbre.fragment_synthesizer,
        timbre_projector=timbre.timbre_projector,
        assembler=build_audio_assembler(),
    )


def build_slider_controls(
    config: HybridConfig, **overrides: Optional[Union[float, bool, dict[str, float]]]
) -> SliderControls:
    allow_full_gibberish = overrides.get("allow_full_gibberish")
    return SliderControls(
        jibberish=float(overrides.get("jibberish", config.sliders["jibberish"].default)),
        clarity=float(overrides.get("clarity", config.sliders["clarity"].default)),
        timing_instability=float(
            overrides.get("timing_instability", config.sliders["timing_instability"].default)
        ),
        allow_full_gibberish=(
            bool(allow_full_gibberish)
            if allow_full_gibberish is not None
            else bool(config.global_constraints.allow_full_gibberish)
        ),
        distance_overrides=dict(overrides.get("distance_overrides", {}) or {}),
    )


def map_severity_to_slider_controls(severity: SeverityProfile) -> SliderControls:
    def clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    jibberish = clamp_unit(
        (severity.global_severity * 0.40)
        + (severity.substitution_bias * 0.35)
        + (severity.addition_bias * 0.25)
    )
    clarity = clamp_unit(
        (severity.distortion_bias * 0.55)
        + (severity.global_severity * 0.25)
        + (severity.complexity_slope * 0.20)
    )
    timing_instability = clamp_unit(
        ((severity.vowel_lengthening_bias + severity.consonant_lengthening_bias) * 0.35)
        + (severity.global_severity * 0.30)
        + (severity.complexity_slope * 0.10)
    )
    return SliderControls(
        jibberish=jibberish,
        clarity=clarity,
        timing_instability=timing_instability,
        allow_full_gibberish=False,
    )


def run_hybrid_acceptance_report(
    config: HybridConfig, audio_path: str
) -> PronunciationAcceptanceReport:
    audio = WavAudioReader().read(audio_path)
    planning = build_hybrid_planning_slice(config)
    timing = build_hybrid_timing_slice(config)
    case_results = []

    for case in config.acceptance_tests:
        target_word = case.word or "yes"
        alignment = planning.aligner.align(audio, target_word)
        graph = planning.feature_extractor.build_phone_graph(alignment)
        controls = build_slider_controls(
            config,
            **case.controls,
            distance_overrides={
                str(key): float(value)
                for key, value in case.evidence.items()
                if isinstance(value, (float, int, bool))
            },
        )
        plan = planning.planner.plan(graph, controls, audio=audio)
        timing_plan = timing.timing_planner.build(graph, plan)
        passed = plan.similarity.passed and timing_plan.preserve_phone_order
        if case.expected == "increase_timing_instability_not_jibberish_or_clarity":
            passed = (
                plan.slider_estimates["timing_instability"].value > plan.slider_estimates["jibberish"].value
                and plan.slider_estimates["timing_instability"].value > plan.slider_estimates["clarity"].value
            )
        case_results.append(
            AcceptanceCaseResult(
                name=case.name,
                word=target_word,
                expected=case.expected,
                passed=passed,
                similarity_score=plan.similarity.score,
                similarity_threshold=plan.similarity.threshold,
                operation_count=len(plan.operations),
                timing_budget_count=len(timing_plan.budgets),
                preserve_phone_order=timing_plan.preserve_phone_order,
            )
        )

    return PronunciationAcceptanceReport(config_name=config.name, case_results=case_results)


def build_speech_distortion_pipeline(
    config: PipelineConfig,
    *,
    hybrid_config: Optional[HybridConfig] = None,
    prefer_pronunciation_safe: bool = False,
) -> SpeechDistortionPipeline:
    stitching = build_stitching_slice(config)
    if hybrid_config is None:
        return SpeechDistortionPipeline(
            aligner=stitching.aligner,
            feature_extractor=stitching.feature_extractor,
            complexity_scorer=stitching.complexity_scorer,
            planner=stitching.planner,
            budget_manager=stitching.budget_manager,
            source_editor=stitching.source_editor,
            fragment_synthesizer=stitching.fragment_synthesizer,
            timbre_projector=stitching.timbre_projector,
            assembler=stitching.assembler,
            prefer_pronunciation_safe=False,
        )

    pronunciation = build_hybrid_stitching_slice(hybrid_config)
    return SpeechDistortionPipeline(
        aligner=stitching.aligner,
        feature_extractor=stitching.feature_extractor,
        complexity_scorer=stitching.complexity_scorer,
        planner=stitching.planner,
        budget_manager=stitching.budget_manager,
        source_editor=stitching.source_editor,
        fragment_synthesizer=stitching.fragment_synthesizer,
        timbre_projector=stitching.timbre_projector,
        assembler=stitching.assembler,
        pronunciation_planner=pronunciation.planner,
        pronunciation_timing_planner=pronunciation.timing_planner,
        pronunciation_source_editor=pronunciation.source_editor,
        pronunciation_fragment_synthesizer=pronunciation.fragment_synthesizer,
        pronunciation_timbre_projector=pronunciation.timbre_projector,
        prefer_pronunciation_safe=prefer_pronunciation_safe,
    )
