from __future__ import annotations

from dataclasses import dataclass

from speech_distortion_pipeline.alignment import Aligner, build_aligner
from speech_distortion_pipeline.config import PipelineConfig
from speech_distortion_pipeline.editing import SourceSegmentEditor, build_source_editor
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
    build_error_planner,
    build_severity_profile_from_config,
)
from speech_distortion_pipeline.resynthesis import (
    FragmentSynthesizer,
    build_fragment_synthesizer,
)
from speech_distortion_pipeline.stitching import AudioAssembler, build_audio_assembler
from speech_distortion_pipeline.timbre import TimbreProjector, build_timbre_projector
from speech_distortion_pipeline.models import SeverityProfile
from speech_distortion_pipeline.timing import DurationBudgetManager, build_duration_budget_manager


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
