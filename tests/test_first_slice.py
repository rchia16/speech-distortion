from pathlib import Path

from speech_distortion_pipeline.alignment import build_aligner
from speech_distortion_pipeline.bootstrap import (
    build_editing_slice,
    build_phonology_slice,
    build_planning_slice,
    build_resynthesis_slice,
    build_stitching_slice,
    build_timbre_slice,
    build_timing_slice,
)
from speech_distortion_pipeline.config import load_config
from speech_distortion_pipeline.io import WavAudioReader
from speech_distortion_pipeline.models import EditType


def test_config_loader_parses_example() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    assert config.sample_rate_hz == 24000
    assert config.alignment.runtime_backend == "torchaudio_ctc"
    assert config.timbre.enabled is True


def test_runtime_aligner_emits_fallback_alignment() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    aligner = build_aligner(config.alignment)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    result = aligner.align(audio, "yes")

    assert result.backend_name == "torchaudio_ctc:fallback_uniform"
    assert len(result.words) == 1
    assert result.words[0].text == "yes"
    assert len(result.phones) == 1


def test_phonology_slice_builds_phone_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_phonology_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "yes")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)

    assert [node.phone for node in graph.nodes] == ["Y", "EH", "S"]
    assert len(graph.word_complexities) == 1
    assert graph.word_complexities[0].word == "yes"
    assert graph.word_complexities[0].score > 0.0


def test_complexity_scores_harder_word_higher() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_phonology_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    easy_graph = runtime.complexity_scorer.score(
        runtime.feature_extractor.build_phone_graph(runtime.aligner.align(audio, "cat"))
    )
    hard_graph = runtime.complexity_scorer.score(
        runtime.feature_extractor.build_phone_graph(runtime.aligner.align(audio, "string"))
    )

    assert hard_graph.word_complexities[0].score > easy_graph.word_complexities[0].score


def test_planning_slice_emits_expected_operation_types() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "rabbit blue string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)

    edit_types = {operation.edit_type for operation in plan.operations}
    assert EditType.SUBSTITUTE in edit_types
    assert EditType.ADD in edit_types
    assert EditType.DISTORT in edit_types
    assert EditType.LENGTHEN_VOWEL in edit_types
    assert EditType.LENGTHEN_CONSONANT in edit_types


def test_planner_marks_distorted_substitution_for_fricative_word() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_planning_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)

    assert any(operation.edit_type == EditType.DISTORTED_SUBSTITUTE for operation in plan.operations)


def test_timing_slice_builds_budgets_for_lengthening_operations() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_timing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "rabbit blue string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)
    timing = runtime.budget_manager.build(graph, plan)

    assert len(timing.budgets) >= 2
    assert timing.total_requested_delta_sec > 0.0
    assert timing.total_compensated_delta_sec > 0.0


def test_timing_budget_window_covers_neighbors() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_timing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)
    timing = runtime.budget_manager.build(graph, plan)

    assert any(budget.window_end_phone_index > budget.window_start_phone_index for budget in timing.budgets)


def test_editing_slice_emits_transformed_audio() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_editing_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "rabbit blue string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)
    timing = runtime.budget_manager.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)

    assert len(edited.samples) == len(audio.samples)
    assert edited.metadata["editor_name"] == "heuristic_source_segment_editor_v1"
    assert edited.samples != audio.samples


def test_resynthesis_slice_emits_fragments_for_symbolic_edits() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_resynthesis_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "rabbit blue string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)
    timing = runtime.budget_manager.build(graph, plan)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)

    assert len(fragments) >= 3
    assert all(fragment.samples for fragment in fragments)
    assert any("substitute" in (fragment.label or "") for fragment in fragments)
    assert any("add" in (fragment.label or "") for fragment in fragments)


def test_stitching_slice_emits_final_audio() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_stitching_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "rabbit blue string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)
    timing = runtime.budget_manager.build(graph, plan)
    edited = runtime.source_editor.apply(audio, graph, plan, timing)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
    projected = runtime.timbre_projector.project(audio, fragments)
    stitched = runtime.assembler.assemble(audio, edited, projected)

    assert len(stitched.samples) >= len(audio.samples)
    assert stitched.metadata["assembler_name"] == "heuristic_audio_assembler_v1"
    assert stitched.metadata["fragment_mix_count"] == str(len(projected))


def test_timbre_slice_projects_fragment_samples() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    runtime = build_timbre_slice(config)
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    alignment = runtime.aligner.align(audio, "rabbit blue string")
    graph = runtime.feature_extractor.build_phone_graph(alignment)
    graph = runtime.complexity_scorer.score(graph)
    plan = runtime.planner.plan(graph, runtime.severity_profile)
    timing = runtime.budget_manager.build(graph, plan)
    fragments = runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
    projected = runtime.timbre_projector.project(audio, fragments)

    assert len(projected) == len(fragments)
    assert all(fragment.source == "heuristic_timbre_projector_v1" for fragment in projected)
    assert any((projected[i].samples or []) != (fragments[i].samples or []) for i in range(len(fragments)))
