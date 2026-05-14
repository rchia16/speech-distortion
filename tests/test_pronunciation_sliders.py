from pathlib import Path

from speech_distortion_pipeline.bootstrap import (
    build_hybrid_editing_slice,
    build_hybrid_planning_slice,
    build_hybrid_resynthesis_slice,
    build_hybrid_stitching_slice,
    build_hybrid_timbre_slice,
    build_hybrid_timing_slice,
    build_slider_controls,
    run_hybrid_acceptance_report,
)
from speech_distortion_pipeline.config import load_hybrid_config
from speech_distortion_pipeline.io import WavAudioReader


def test_pronunciation_slider_loader_parses_repo_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")

    assert config.version == 4
    assert config.global_constraints.minimum_pronunciation_similarity == 0.72
    assert config.pronunciation_skeleton.anchor_selection_rules.short_word_anchor_all_phones_when_phone_count_lte == 3
    assert "jibberish" in config.sliders
    assert config.sliders["jibberish"].guardrails.short_word_mode is not None


def test_pronunciation_planner_anchors_all_phones_for_short_word() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "no")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(graph, build_slider_controls(config, jibberish=1.0), audio=audio)

    assert len(plan.skeletons) == 1
    skeleton = plan.skeletons[0]
    assert skeleton.phone_sequence == ["N", "OW"]
    assert skeleton.anchor_phone_indices == skeleton.phone_indices
    assert plan.similarity.passed is True
    assert plan.traversal_paths
    assert any(path.mode in {"symbolic_first", "hubert_first"} for path in plan.traversal_paths)


def test_high_clarity_keeps_symbolic_plan_empty() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "stop")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(
        graph,
        build_slider_controls(config, jibberish=0.0, clarity=1.0, timing_instability=0.0),
        audio=audio,
    )

    assert plan.operations == []
    assert plan.clarity_plan.parameters["fricative_smearing"] > 0.0
    assert plan.similarity.score >= plan.similarity.threshold
    assert plan.slider_estimates["clarity"].value > plan.slider_estimates["jibberish"].value
    assert plan.trace is not None


def test_pronunciation_timing_preserves_order_and_caps_repeats() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_timing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "yes")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, timing_instability=1.0)
    plan = runtime.planner.plan(graph, controls, audio=audio)
    timing = runtime.timing_planner.build(graph, plan)

    assert timing.preserve_phone_order is True
    assert len(timing.onset_repeat_phone_indices) <= config.sliders["timing_instability"].guardrails.values["max_onset_repeats"]
    assert timing.total_requested_delta_sec >= 0.0
    assert timing.pause_after_phone_indices


def test_full_gibberish_override_lowers_similarity_threshold() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "no")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    normal = runtime.planner.plan(graph, build_slider_controls(config, jibberish=1.0), audio=audio)
    override = runtime.planner.plan(
        graph,
        build_slider_controls(config, jibberish=1.0, allow_full_gibberish=True),
        audio=audio,
    )

    assert override.similarity.threshold < normal.similarity.threshold
    assert override.similarity.threshold == 0.25


def test_pronunciation_acceptance_report_runs_all_configured_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    report = run_hybrid_acceptance_report(config, str(root / "yes_slow.wav"))

    assert report.config_name == config.name
    assert len(report.case_results) == len(config.acceptance_tests)
    assert report.passed is True
    assert all(
        case.similarity_score >= case.similarity_threshold
        or case.expected == "increase_timing_instability_not_jibberish_or_clarity"
        for case in report.case_results
    )


def test_dangerous_grapheme_candidate_is_rejected_by_similarity() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "bath")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(graph, build_slider_controls(config, jibberish=1.0), audio=audio)

    bathe_candidates = [candidate for candidate in plan.candidates if candidate.grapheme == "bathe"]
    assert not bathe_candidates or all((not candidate.accepted) for candidate in bathe_candidates)


def test_temporal_distance_routes_to_timing_slider() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "yes")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(
        graph,
        build_slider_controls(
            config,
            distance_overrides={
                "temporal_distance": 1.0,
                "dtw_warp_distance": 0.9,
                "phoneme_distance": 0.0,
                "acoustic_embedding_distance": 0.0,
            },
        ),
        audio=audio,
    )

    assert plan.slider_estimates["timing_instability"].value > plan.slider_estimates["jibberish"].value
    assert plan.slider_estimates["timing_instability"].value > plan.slider_estimates["clarity"].value
    assert plan.slider_estimates["timing_instability"].routing_rule == "temporal_mismatch"


def test_pronunciation_editing_stage_emits_transformed_audio_and_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_editing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "yes")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, clarity=0.8, timing_instability=0.9)
    plan = runtime.planner.plan(graph, controls, audio=audio)
    timing = runtime.timing_planner.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)

    assert len(edited.samples) == len(audio.samples)
    assert edited.samples != audio.samples
    assert edited.metadata["editor_name"] == "heuristic_pronunciation_source_editor_v1"
    assert edited.metadata["timing_pause_count"] != "0"
    assert float(edited.metadata["pronunciation_similarity_score"]) >= plan.similarity.threshold
    assert edited.metadata["trace_selected_grapheme_variant"] == plan.trace.selected_grapheme_variant
    assert plan.trace.routing_rules["timing_instability"]


def test_pronunciation_resynthesis_stage_emits_fragments_for_symbolic_edits() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_resynthesis_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=0.8, clarity=0.5, timing_instability=0.7)
    plan = runtime.planner.plan(graph, controls, audio=audio)
    timing = runtime.timing_planner.build(graph, plan)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)

    assert fragments
    assert all(fragment.samples for fragment in fragments)
    assert all(
        fragment.source in {
            "heuristic_pronunciation_fragment_synthesizer_v1",
            "kokoro_pronunciation_fragment_synthesizer_v1",
            "system_say_fragment_synthesizer_v1",
            "coqui_tts_fragment_synthesizer_v1",
        }
        for fragment in fragments
    )
    assert any(
        "substitute" in (fragment.label or "")
        or "distort" in (fragment.label or "")
        or "word_variant:" in (fragment.label or "")
        for fragment in fragments
    )


def test_pronunciation_resynthesis_prefers_word_variant_fragment_for_safe_candidate() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_resynthesis_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "no")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=1.0, clarity=0.2, timing_instability=0.0)
    plan = runtime.planner.plan(graph, controls, audio=audio)
    timing = runtime.timing_planner.build(graph, plan)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)

    mutated_candidates = [
        candidate for candidate in plan.selected_candidates if candidate.accepted and candidate.grapheme != candidate.word
    ]
    if mutated_candidates:
        assert any(
            (fragment.label or "").startswith("word_variant:{0}".format(mutated_candidates[0].grapheme))
            for fragment in fragments
        )


def test_pronunciation_timbre_stage_projects_fragments() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_timbre_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=0.8, clarity=0.5, timing_instability=0.7)
    plan = runtime.planner.plan(graph, controls, audio=audio)
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
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_stitching_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    controls = build_slider_controls(config, jibberish=0.8, clarity=0.5, timing_instability=0.7)
    plan = runtime.planner.plan(graph, controls, audio=audio)
    timing = runtime.timing_planner.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
    projected = runtime.timbre_projector.project(audio, fragments)
    stitched = runtime.assembler.assemble(audio, edited, projected)

    assert len(stitched.samples) >= len(audio.samples)
    assert stitched.metadata["assembler_name"] == "heuristic_audio_assembler_v1"
    assert stitched.metadata["fragment_mix_count"] == str(len(projected))


def test_plan_exposes_distance_features_and_routing() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(
        graph,
        build_slider_controls(config, jibberish=0.8, clarity=0.4, timing_instability=0.7),
        audio=audio,
    )

    assert plan.evidence_bundle is not None
    assert "phoneme_distance" in plan.distance_evidence
    assert plan.distance_evidence["phoneme_distance"].raw_features
    assert plan.trace is not None
    assert "jibberish" in plan.trace.routing_rules
    assert "acoustic_backend" in plan.distance_evidence["acoustic_embedding_distance"].raw_features
    assert plan.trace.acoustic_model_compatibility
    assert plan.trace.traversal_source in {"symbolic_first", "hubert_first", "grapheme_fallback", ""}
    assert isinstance(plan.trace.traversal_path, list)
    contextual_paths = [path for path in plan.traversal_paths if path.mode in {"symbolic_first", "hubert_first"}]
    assert contextual_paths
    if any(path.mode == "hubert_first" for path in contextual_paths):
        assert any(path.proposal_source == "hubert_reference_bank" for path in contextual_paths if path.mode == "hubert_first")
        assert any(path.reference_neighbors for path in contextual_paths if path.mode == "hubert_first")


def test_acoustic_backend_metadata_exposes_proxy_fallback_or_loaded_model() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "hybrid.yaml")
    runtime = build_hybrid_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "stop")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    plan = runtime.planner.plan(
        graph,
        build_slider_controls(config, clarity=0.7),
        audio=audio,
    )

    acoustic_features = plan.distance_evidence["acoustic_embedding_distance"].raw_features
    compatibility = str(acoustic_features.get("acoustic_model_compatibility", ""))
    assert acoustic_features.get("acoustic_backend") in {"proxy", "torchaudio_hubert"}
    assert acoustic_features.get("acoustic_model_path")
    assert compatibility
    assert compatibility in {
        "fairseq_checkpoint_incompatible",
        "torchaudio_state_dict_loaded",
        "torchaudio_state_dict_incompatible",
        "checkpoint_missing",
        "checkpoint_load_failed",
        "unsupported_checkpoint_payload",
        "torchaudio_unavailable",
    }
    assert "temporal_distance" in plan.trace.distance_features
