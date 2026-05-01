from __future__ import annotations

from dataclasses import dataclass

from speech_distortion_pipeline.alignment import Aligner
from speech_distortion_pipeline.editing import SourceSegmentEditor
from speech_distortion_pipeline.models import AudioBuffer, EditPlan, SeverityProfile
from speech_distortion_pipeline.phonology import ComplexityScorer, FeatureExtractor
from speech_distortion_pipeline.planning import ErrorPlanner
from speech_distortion_pipeline.resynthesis import FragmentSynthesizer
from speech_distortion_pipeline.stitching import AudioAssembler
from speech_distortion_pipeline.timbre import TimbreProjector
from speech_distortion_pipeline.timing import DurationBudgetManager


@dataclass
class SpeechDistortionPipeline:
    aligner: Aligner
    feature_extractor: FeatureExtractor
    complexity_scorer: ComplexityScorer
    planner: ErrorPlanner
    budget_manager: DurationBudgetManager
    source_editor: SourceSegmentEditor
    fragment_synthesizer: FragmentSynthesizer
    timbre_projector: TimbreProjector
    assembler: AudioAssembler

    def run(self, audio: AudioBuffer, transcript: str, severity: SeverityProfile) -> AudioBuffer:
        """End-to-end entry point for the full speech distortion pipeline."""
        raise NotImplementedError("Pipeline orchestration is not implemented yet.")

    def plan_only(self, audio: AudioBuffer, transcript: str, severity: SeverityProfile) -> EditPlan:
        """Debug entry point that emits the intermediate edit plan without rendering."""
        raise NotImplementedError("Planner-only orchestration is not implemented yet.")
