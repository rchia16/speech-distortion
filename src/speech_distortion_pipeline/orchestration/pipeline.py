from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from speech_distortion_pipeline.alignment import Aligner
from speech_distortion_pipeline.editing import PronunciationSourceEditor, SourceSegmentEditor
from speech_distortion_pipeline.models import (
    AudioBuffer,
    AudioSegment,
    EditPlan,
    PronunciationSafePlan,
    SeverityProfile,
    TimingPlan,
    TimingInstabilityPlan,
)
from speech_distortion_pipeline.phonology import ComplexityScorer, FeatureExtractor
from speech_distortion_pipeline.planning import ErrorPlanner, HybridPlanner
from speech_distortion_pipeline.resynthesis import FragmentSynthesizer, PronunciationFragmentSynthesizer
from speech_distortion_pipeline.stitching import AudioAssembler
from speech_distortion_pipeline.timbre import TimbreProjector
from speech_distortion_pipeline.timing import DurationBudgetManager, PronunciationTimingPlanner
from speech_distortion_pipeline.models import SliderControls


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
    pronunciation_planner: Optional[HybridPlanner] = None
    pronunciation_timing_planner: Optional[PronunciationTimingPlanner] = None
    pronunciation_source_editor: Optional[PronunciationSourceEditor] = None
    pronunciation_fragment_synthesizer: Optional[PronunciationFragmentSynthesizer] = None
    pronunciation_timbre_projector: Optional[TimbreProjector] = None
    prefer_pronunciation_safe: bool = False

    def run(self, audio: AudioBuffer, transcript: str, severity: SeverityProfile) -> AudioBuffer:
        """End-to-end entry point for the full speech distortion pipeline."""
        if self.prefer_pronunciation_safe and self._has_pronunciation_runtime():
            return self._run_pronunciation_safe(audio, transcript, severity)

        alignment = self.aligner.align(audio, transcript)
        graph = self.feature_extractor.build_phone_graph(alignment)
        graph = self.complexity_scorer.score(graph)
        plan = self.planner.plan(graph, severity)
        timing = self.budget_manager.build(graph, plan)
        edited = self.source_editor.apply(audio, graph, plan, timing)
        fragments = self.fragment_synthesizer.synthesize(audio, graph, plan, timing)
        projected = self.timbre_projector.project(audio, fragments)
        return self.assembler.assemble(audio, edited, projected)

    def plan_only(self, audio: AudioBuffer, transcript: str, severity: SeverityProfile) -> EditPlan:
        """Debug entry point that emits the intermediate edit plan without rendering."""
        alignment = self.aligner.align(audio, transcript)
        graph = self.feature_extractor.build_phone_graph(alignment)
        if self.prefer_pronunciation_safe and self._has_pronunciation_runtime():
            pronunciation_plan = self._plan_pronunciation_safe(graph, severity, audio=audio)
            return EditPlan(
                operations=list(pronunciation_plan.operations),
                planner_name="migration_pronunciation_safe_bridge",
            )

        graph = self.complexity_scorer.score(graph)
        return self.planner.plan(graph, severity)

    def timing_only(
        self, audio: AudioBuffer, transcript: str, severity: SeverityProfile
    ) -> TimingPlan | TimingInstabilityPlan:
        alignment = self.aligner.align(audio, transcript)
        graph = self.feature_extractor.build_phone_graph(alignment)
        if self.prefer_pronunciation_safe and self._has_pronunciation_runtime():
            assert self.pronunciation_timing_planner is not None
            plan = self._plan_pronunciation_safe(graph, severity, audio=audio)
            return self.pronunciation_timing_planner.build(graph, plan)

        graph = self.complexity_scorer.score(graph)
        plan = self.planner.plan(graph, severity)
        return self.budget_manager.build(graph, plan)

    def edit_only(self, audio: AudioBuffer, transcript: str, severity: SeverityProfile) -> AudioBuffer:
        alignment = self.aligner.align(audio, transcript)
        graph = self.feature_extractor.build_phone_graph(alignment)
        if self.prefer_pronunciation_safe and self._has_pronunciation_runtime():
            assert self.pronunciation_timing_planner is not None
            assert self.pronunciation_source_editor is not None
            plan = self._plan_pronunciation_safe(graph, severity, audio=audio)
            timing = self.pronunciation_timing_planner.build(graph, plan)
            return self.pronunciation_source_editor.apply(audio, graph, plan, timing)

        graph = self.complexity_scorer.score(graph)
        plan = self.planner.plan(graph, severity)
        timing = self.budget_manager.build(graph, plan)
        return self.source_editor.apply(audio, graph, plan, timing)

    def resynthesize_only(
        self, audio: AudioBuffer, transcript: str, severity: SeverityProfile
    ) -> list[AudioSegment]:
        alignment = self.aligner.align(audio, transcript)
        graph = self.feature_extractor.build_phone_graph(alignment)
        if self.prefer_pronunciation_safe and self._has_pronunciation_runtime():
            assert self.pronunciation_timing_planner is not None
            assert self.pronunciation_fragment_synthesizer is not None
            plan = self._plan_pronunciation_safe(graph, severity, audio=audio)
            timing = self.pronunciation_timing_planner.build(graph, plan)
            return self.pronunciation_fragment_synthesizer.synthesize(audio, graph, plan, timing)

        graph = self.complexity_scorer.score(graph)
        plan = self.planner.plan(graph, severity)
        timing = self.budget_manager.build(graph, plan)
        return self.fragment_synthesizer.synthesize(audio, graph, plan, timing)

    def timbre_only(
        self, audio: AudioBuffer, transcript: str, severity: SeverityProfile
    ) -> list[AudioSegment]:
        if self.prefer_pronunciation_safe and self._has_pronunciation_runtime():
            assert self.pronunciation_timbre_projector is not None
            fragments = self.resynthesize_only(audio, transcript, severity)
            return self.pronunciation_timbre_projector.project(audio, fragments)

        fragments = self.resynthesize_only(audio, transcript, severity)
        return self.timbre_projector.project(audio, fragments)

    def _has_pronunciation_runtime(self) -> bool:
        return (
            self.pronunciation_planner is not None
            and self.pronunciation_timing_planner is not None
            and self.pronunciation_source_editor is not None
            and self.pronunciation_fragment_synthesizer is not None
            and self.pronunciation_timbre_projector is not None
        )

    def _plan_pronunciation_safe(
        self,
        graph,
        severity: SeverityProfile,
        audio: Optional[AudioBuffer] = None,
    ) -> PronunciationSafePlan:
        assert self.pronunciation_planner is not None
        controls = self._map_severity_to_slider_controls(severity)
        return self.pronunciation_planner.plan(graph, controls, audio=audio)

    def _run_pronunciation_safe(
        self, audio: AudioBuffer, transcript: str, severity: SeverityProfile
    ) -> AudioBuffer:
        assert self.pronunciation_timing_planner is not None
        assert self.pronunciation_source_editor is not None
        assert self.pronunciation_fragment_synthesizer is not None
        assert self.pronunciation_timbre_projector is not None

        alignment = self.aligner.align(audio, transcript)
        graph = self.feature_extractor.build_phone_graph(alignment)
        plan = self._plan_pronunciation_safe(graph, severity, audio=audio)
        timing = self.pronunciation_timing_planner.build(graph, plan)
        edited = self.pronunciation_source_editor.apply(audio, graph, plan, timing)
        fragments = self.pronunciation_fragment_synthesizer.synthesize(audio, graph, plan, timing)
        projected = self.pronunciation_timbre_projector.project(audio, fragments)
        stitched = self.assembler.assemble(audio, edited, projected)
        stitched.metadata.update(
            {
                "migration_mode": "hybrid_distance_aware_runtime",
                "mapped_jibberish": f"{plan.controls.jibberish:.3f}",
                "mapped_clarity": f"{plan.controls.clarity:.3f}",
                "mapped_timing_instability": f"{plan.controls.timing_instability:.3f}",
                "pronunciation_similarity_score": f"{plan.similarity.score:.4f}",
                "pronunciation_similarity_threshold": f"{plan.similarity.threshold:.4f}",
                "trace_selected_grapheme_variant": (
                    plan.trace.selected_grapheme_variant if plan.trace is not None else ""
                ),
                "trace_distance_evidence_keys": ",".join(sorted(plan.distance_evidence.keys())),
            }
        )
        return stitched

    def _map_severity_to_slider_controls(self, severity: SeverityProfile) -> SliderControls:
        def clamp_unit(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        return SliderControls(
            jibberish=clamp_unit(
                (severity.global_severity * 0.40)
                + (severity.substitution_bias * 0.35)
                + (severity.addition_bias * 0.25)
            ),
            clarity=clamp_unit(
                (severity.distortion_bias * 0.55)
                + (severity.global_severity * 0.25)
                + (severity.complexity_slope * 0.20)
            ),
            timing_instability=clamp_unit(
                ((severity.vowel_lengthening_bias + severity.consonant_lengthening_bias) * 0.35)
                + (severity.global_severity * 0.30)
                + (severity.complexity_slope * 0.10)
            ),
            allow_full_gibberish=False,
        )
