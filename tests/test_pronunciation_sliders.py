from pathlib import Path

from speech_distortion_pipeline.bootstrap import (
    build_pronunciation_editing_slice,
    build_pronunciation_planning_slice,
    build_pronunciation_resynthesis_slice,
    build_pronunciation_stitching_slice,
    build_pronunciation_timbre_slice,
    build_pronunciation_timing_slice,
    build_slider_controls,
    run_pronunciation_acceptance_report,
)
from speech_distortion_pipeline.config import load_pronunciation_slider_config
from speech_distortion_pipeline.io import WavAudioReader


def test_pronunciation_slider_loader_parses_repo_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")

    assert config.version == 2
    assert config.global_constraints.minimum_pronunciation_similarity == 0.72
    assert config.pronunciation_skeleton.anchor_selection_rules.short_word_anchor_all_phones_when_phone_count_lte == 3
    assert "jibberish" in config.sliders
    assert config.sliders["jibberish"].pronunciation_guardrails.short_word_mode is not None


def test_pronunciation_planner_anchors_all_phones_for_short_word() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "no")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(graph, build_slider_controls(config, jibberish=1.0))

    assert len(plan.skeletons) == 1
    skeleton = plan.skeletons[0]
    assert skeleton.phone_sequence == ["N", "OW"]
    assert skeleton.anchor_phone_indices == skeleton.phone_indices
    assert plan.similarity.passed is True


def test_high_clarity_keeps_symbolic_plan_empty() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "stop")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(
        graph,
        build_slider_controls(config, jibberish=0.0, clarity=1.0, timing_instability=0.0),
    )

    assert plan.operations == []
    assert plan.clarity_plan.parameters["fricative_smearing"] > 0.0
    assert plan.similarity.score >= plan.similarity.threshold


def test_pronunciation_timing_preserves_order_and_caps_repeats() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_timing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "yes")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, timing_instability=1.0)
    plan = runtime.planner.plan(graph, controls)
    timing = runtime.timing_planner.build(graph, plan)

    assert timing.preserve_phone_order is True
    assert len(timing.onset_repeat_phone_indices) <= config.sliders["timing_instability"].pronunciation_guardrails.values["max_onset_repeats"]
    assert timing.total_requested_delta_sec >= 0.0
    assert timing.pause_after_phone_indices


def test_full_gibberish_override_lowers_similarity_threshold() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "no")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    normal = runtime.planner.plan(graph, build_slider_controls(config, jibberish=1.0))
    override = runtime.planner.plan(
        graph,
        build_slider_controls(config, jibberish=1.0, allow_full_gibberish=True),
    )

    assert override.similarity.threshold < normal.similarity.threshold
    assert override.similarity.threshold == 0.25


def test_pronunciation_acceptance_report_runs_all_configured_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    report = run_pronunciation_acceptance_report(config, str(root / "yes_slow.wav"))

    assert report.config_name == config.name
    assert len(report.case_results) == len(config.acceptance_tests)
    assert report.passed is True
    assert all(case.similarity_score >= case.similarity_threshold for case in report.case_results)


def test_pronunciation_editing_stage_emits_transformed_audio_and_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_editing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "yes")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, clarity=0.8, timing_instability=0.9)
    plan = runtime.planner.plan(graph, controls)
    timing = runtime.timing_planner.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)

    assert len(edited.samples) == len(audio.samples)
    assert edited.samples != audio.samples
    assert edited.metadata["editor_name"] == "heuristic_pronunciation_source_editor_v1"
    assert edited.metadata["timing_pause_count"] != "0"
    assert float(edited.metadata["pronunciation_similarity_score"]) >= plan.similarity.threshold


def test_pronunciation_resynthesis_stage_emits_fragments_for_symbolic_edits() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_resynthesis_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=0.8, clarity=0.5, timing_instability=0.7)
    plan = runtime.planner.plan(graph, controls)
    timing = runtime.timing_planner.build(graph, plan)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)

    assert fragments
    assert all(fragment.samples for fragment in fragments)
    assert all(fragment.source == "heuristic_pronunciation_fragment_synthesizer_v1" for fragment in fragments)
    assert any("substitute" in (fragment.label or "") or "distort" in (fragment.label or "") for fragment in fragments)


def test_pronunciation_timbre_stage_projects_fragments() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_timbre_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=0.8, clarity=0.5, timing_instability=0.7)
    plan = runtime.planner.plan(graph, controls)
    timing = runtime.timing_planner.build(graph, plan)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
    projected = runtime.timbre_projector.project(audio, fragments)

    assert len(projected) == len(fragments)
    assert all(fragment.source == "heuristic_timbre_projector_v1" for fragment in projected)
    assert [fragment.label for fragment in projected] == [fragment.label for fragment in fragments]
    assert [fragment.start_sec for fragment in projected] == [fragment.start_sec for fragment in fragments]
    assert any((projected[i].samples or []) != (fragments[i].samples or []) for i in range(len(fragments)))


def test_pronunciation_stitching_stage_emits_final_audio_with_fragments() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pronunciation_slider_config(root / "pronunciation_sliders.yaml")
    runtime = build_pronunciation_stitching_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=0.8, clarity=0.5, timing_instability=0.7)
    plan = runtime.planner.plan(graph, controls)
    timing = runtime.timing_planner.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
    projected = runtime.timbre_projector.project(audio, fragments)
    stitched = runtime.assembler.assemble(audio, edited, projected)

    assert len(stitched.samples) >= len(audio.samples)
    assert stitched.metadata["assembler_name"] == "heuristic_audio_assembler_v1"
    assert stitched.metadata["fragment_mix_count"] == str(len(projected))
