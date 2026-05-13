from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from speech_distortion_pipeline.bootstrap import (
    build_slider_controls,
    build_speech_distortion_pipeline,
    build_hybrid_editing_slice,
    build_hybrid_planning_slice,
    build_hybrid_resynthesis_slice,
    build_hybrid_stitching_slice,
    build_hybrid_timbre_slice,
    build_hybrid_timing_slice,
    run_hybrid_acceptance_report,
    build_editing_slice,
    build_first_slice,
    build_phonology_slice,
    build_planning_slice,
    build_resynthesis_slice,
    build_stitching_slice,
    build_timbre_slice,
    build_timing_slice,
)
from speech_distortion_pipeline.config import load_config, load_hybrid_config
from speech_distortion_pipeline.io import WavAudioReader, WavAudioWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bootstrap speech distortion slices.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    parser.add_argument("--audio", required=True, help="Path to input WAV file.")
    parser.add_argument("--transcript", required=True, help="Transcript text to align.")
    parser.add_argument(
        "--stage",
        choices=(
            "alignment",
            "phonology",
            "planning",
            "timing",
            "editing",
            "resynthesis",
            "timbre",
            "stitching",
            "hybrid_planning",
            "hybrid_timing",
            "hybrid_editing",
            "hybrid_resynthesis",
            "hybrid_timbre",
            "hybrid_stitching",
            "hybrid_acceptance",
        ),
        default="stitching",
        help="Bootstrap stage to run.",
    )
    parser.add_argument("--output", help="Output WAV path for editing or stitching stage.")
    parser.add_argument(
        "--output-dir",
        help="Output directory for resynthesis fragments.",
    )
    parser.add_argument("--jibberish", type=float, help="Override jibberish slider value.")
    parser.add_argument("--clarity", type=float, help="Override clarity slider value.")
    parser.add_argument("--timing-instability", type=float, help="Override timing instability slider value.")
    parser.add_argument(
        "--allow-full-gibberish",
        action="store_true",
        help="Enable the full-gibberish override for hybrid slider stages.",
    )
    parser.add_argument(
        "--prefer-hybrid-runtime",
        action="store_true",
        help="Route planning/stitching stages through the hybrid runtime.",
    )
    parser.add_argument(
        "--hybrid-config",
        default="hybrid.yaml",
        help="Hybrid config path used when --prefer-hybrid-runtime is enabled.",
    )
    args = parser.parse_args()

    audio = WavAudioReader().read(args.audio)
    if args.stage == "hybrid_planning":
        config = load_hybrid_config(args.config)
        slice_runtime = build_hybrid_planning_slice(config)
        controls = build_slider_controls(
            config,
            jibberish=args.jibberish,
            clarity=args.clarity,
            timing_instability=args.timing_instability,
            allow_full_gibberish=args.allow_full_gibberish or None,
        )
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        payload = slice_runtime.planner.plan(graph, controls, audio=audio)
        print(json.dumps(asdict(payload), indent=2))
        return
    elif args.stage == "hybrid_timing":
        config = load_hybrid_config(args.config)
        slice_runtime = build_hybrid_timing_slice(config)
        controls = build_slider_controls(
            config,
            jibberish=args.jibberish,
            clarity=args.clarity,
            timing_instability=args.timing_instability,
            allow_full_gibberish=args.allow_full_gibberish or None,
        )
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        plan = slice_runtime.planner.plan(graph, controls, audio=audio)
        payload = slice_runtime.timing_planner.build(graph, plan)
        print(json.dumps(asdict(payload), indent=2))
        return
    elif args.stage == "hybrid_editing":
        config = load_hybrid_config(args.config)
        slice_runtime = build_hybrid_editing_slice(config)
        controls = build_slider_controls(
            config,
            jibberish=args.jibberish,
            clarity=args.clarity,
            timing_instability=args.timing_instability,
            allow_full_gibberish=args.allow_full_gibberish or None,
        )
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        plan = slice_runtime.planner.plan(graph, controls, audio=audio)
        timing = slice_runtime.timing_planner.build(graph, plan)
        edited = slice_runtime.source_editor.apply(audio, graph, plan, timing)
        output_path = args.output or "hybrid_edited_output.wav"
        WavAudioWriter().write(output_path, edited)
        payload = {
            "output_path": output_path,
            "sample_rate_hz": edited.sample_rate_hz,
            "sample_count": len(edited.samples),
            "metadata": edited.metadata,
            "similarity_score": plan.similarity.score,
            "similarity_threshold": plan.similarity.threshold,
        }
        print(json.dumps(payload, indent=2))
        return
    elif args.stage == "hybrid_resynthesis":
        config = load_hybrid_config(args.config)
        slice_runtime = build_hybrid_resynthesis_slice(config)
        controls = build_slider_controls(
            config,
            jibberish=args.jibberish,
            clarity=args.clarity,
            timing_instability=args.timing_instability,
            allow_full_gibberish=args.allow_full_gibberish or None,
        )
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        plan = slice_runtime.planner.plan(graph, controls, audio=audio)
        timing = slice_runtime.timing_planner.build(graph, plan)
        fragments = slice_runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
        payload = {
            "fragment_count": len(fragments),
            "fragments": [asdict(fragment) for fragment in fragments],
            "similarity_score": plan.similarity.score,
            "similarity_threshold": plan.similarity.threshold,
        }
        print(json.dumps(payload, indent=2))
        return
    elif args.stage == "hybrid_timbre":
        config = load_hybrid_config(args.config)
        slice_runtime = build_hybrid_timbre_slice(config)
        controls = build_slider_controls(
            config,
            jibberish=args.jibberish,
            clarity=args.clarity,
            timing_instability=args.timing_instability,
            allow_full_gibberish=args.allow_full_gibberish or None,
        )
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        plan = slice_runtime.planner.plan(graph, controls, audio=audio)
        timing = slice_runtime.timing_planner.build(graph, plan)
        fragments = slice_runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
        projected = slice_runtime.timbre_projector.project(audio, fragments)
        payload = {
            "projected_fragment_count": len(projected),
            "projected_fragments": [asdict(fragment) for fragment in projected],
            "similarity_score": plan.similarity.score,
            "similarity_threshold": plan.similarity.threshold,
        }
        print(json.dumps(payload, indent=2))
        return
    elif args.stage == "hybrid_stitching":
        config = load_hybrid_config(args.config)
        slice_runtime = build_hybrid_stitching_slice(config)
        controls = build_slider_controls(
            config,
            jibberish=args.jibberish,
            clarity=args.clarity,
            timing_instability=args.timing_instability,
            allow_full_gibberish=args.allow_full_gibberish or None,
        )
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        plan = slice_runtime.planner.plan(graph, controls, audio=audio)
        timing = slice_runtime.timing_planner.build(graph, plan)
        edited = slice_runtime.source_editor.apply(audio, graph, plan, timing)
        fragments = slice_runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
        projected = slice_runtime.timbre_projector.project(audio, fragments)
        stitched = slice_runtime.assembler.assemble(audio, edited, projected)
        output_path = args.output or "hybrid_stitched_output.wav"
        WavAudioWriter().write(output_path, stitched)
        payload = {
            "output_path": output_path,
            "sample_rate_hz": stitched.sample_rate_hz,
            "sample_count": len(stitched.samples),
            "metadata": stitched.metadata,
            "fragment_count": len(projected),
            "projected_fragments": len(projected),
            "similarity_score": plan.similarity.score,
            "similarity_threshold": plan.similarity.threshold,
        }
        print(json.dumps(payload, indent=2))
        return
    elif args.stage == "hybrid_acceptance":
        config = load_hybrid_config(args.config)
        payload = run_hybrid_acceptance_report(config, args.audio)
        print(json.dumps({**asdict(payload), "passed": payload.passed}, indent=2))
        return

    config = load_config(args.config)
    legacy_severity = build_planning_slice(config).severity_profile
    migration_pipeline = None
    if args.prefer_hybrid_runtime:
        hybrid_config = load_hybrid_config(args.hybrid_config)
        migration_pipeline = build_speech_distortion_pipeline(
            config,
            hybrid_config=hybrid_config,
            prefer_pronunciation_safe=True,
        )
    if args.stage == "alignment":
        slice_runtime = build_first_slice(config)
        payload = slice_runtime.aligner.align(audio, args.transcript)
    elif args.stage == "phonology":
        slice_runtime = build_phonology_slice(config)
        alignment = slice_runtime.aligner.align(audio, args.transcript)
        graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
        payload = slice_runtime.complexity_scorer.score(graph)
    elif args.stage == "planning":
        if migration_pipeline is not None:
            payload = migration_pipeline.plan_only(audio, args.transcript, legacy_severity)
        else:
            slice_runtime = build_planning_slice(config)
            alignment = slice_runtime.aligner.align(audio, args.transcript)
            graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
            graph = slice_runtime.complexity_scorer.score(graph)
            payload = slice_runtime.planner.plan(graph, slice_runtime.severity_profile)
    elif args.stage == "timing":
        if migration_pipeline is not None:
            payload = migration_pipeline.timing_only(audio, args.transcript, legacy_severity)
        else:
            slice_runtime = build_timing_slice(config)
            alignment = slice_runtime.aligner.align(audio, args.transcript)
            graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
            graph = slice_runtime.complexity_scorer.score(graph)
            plan = slice_runtime.planner.plan(graph, slice_runtime.severity_profile)
            payload = slice_runtime.budget_manager.build(graph, plan)
        print(json.dumps(asdict(payload), indent=2))
        return
    elif args.stage == "editing":
        if migration_pipeline is not None:
            edited = migration_pipeline.edit_only(audio, args.transcript, legacy_severity)
        else:
            slice_runtime = build_editing_slice(config)
            alignment = slice_runtime.aligner.align(audio, args.transcript)
            graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
            graph = slice_runtime.complexity_scorer.score(graph)
            plan = slice_runtime.planner.plan(graph, slice_runtime.severity_profile)
            timing = slice_runtime.budget_manager.build(graph, plan)
            edited = slice_runtime.source_editor.apply(audio, graph, plan, timing)
        output_path = args.output or "edited_output.wav"
        WavAudioWriter().write(output_path, edited)
        payload = {
            "output_path": output_path,
            "sample_rate_hz": edited.sample_rate_hz,
            "sample_count": len(edited.samples),
            "metadata": edited.metadata,
        }
        print(json.dumps(payload, indent=2))
        return
    elif args.stage == "resynthesis":
        if migration_pipeline is not None:
            fragments = migration_pipeline.resynthesize_only(audio, args.transcript, legacy_severity)
        else:
            slice_runtime = build_resynthesis_slice(config)
            alignment = slice_runtime.aligner.align(audio, args.transcript)
            graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
            graph = slice_runtime.complexity_scorer.score(graph)
            plan = slice_runtime.planner.plan(graph, slice_runtime.severity_profile)
            timing = slice_runtime.budget_manager.build(graph, plan)
            fragments = slice_runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)

        output_dir = args.output_dir or "resynthesized_fragments"
        writer = WavAudioWriter()
        manifest = []
        for index, fragment in enumerate(fragments):
            fragment_audio = audio.__class__(
                samples=fragment.samples or [],
                sample_rate_hz=audio.sample_rate_hz,
                channel_count=1,
                metadata={"fragment_index": str(index), "label": fragment.label or ""},
            )
            path = "{0}/fragment_{1:03d}.wav".format(output_dir, index)
            writer.write(path, fragment_audio)
            manifest.append(
                {
                    "path": path,
                    "start_sec": fragment.start_sec,
                    "end_sec": fragment.end_sec,
                    "label": fragment.label,
                    "sample_count": len(fragment.samples or []),
                    "source": fragment.source,
                }
            )
        payload = {"fragment_count": len(fragments), "output_dir": output_dir, "fragments": manifest}
        print(json.dumps(payload, indent=2))
        return
    elif args.stage == "timbre":
        if migration_pipeline is not None:
            projected = migration_pipeline.timbre_only(audio, args.transcript, legacy_severity)
        else:
            slice_runtime = build_timbre_slice(config)
            alignment = slice_runtime.aligner.align(audio, args.transcript)
            graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
            graph = slice_runtime.complexity_scorer.score(graph)
            plan = slice_runtime.planner.plan(graph, slice_runtime.severity_profile)
            timing = slice_runtime.budget_manager.build(graph, plan)
            fragments = slice_runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
            projected = slice_runtime.timbre_projector.project(audio, fragments)

        output_dir = args.output_dir or "projected_fragments"
        writer = WavAudioWriter()
        manifest = []
        for index, fragment in enumerate(projected):
            fragment_audio = audio.__class__(
                samples=fragment.samples or [],
                sample_rate_hz=audio.sample_rate_hz,
                channel_count=1,
                metadata={"fragment_index": str(index), "label": fragment.label or "", "source": fragment.source},
            )
            path = "{0}/fragment_{1:03d}.wav".format(output_dir, index)
            writer.write(path, fragment_audio)
            manifest.append(
                {
                    "path": path,
                    "start_sec": fragment.start_sec,
                    "end_sec": fragment.end_sec,
                    "label": fragment.label,
                    "sample_count": len(fragment.samples or []),
                    "source": fragment.source,
                }
            )
        payload = {"fragment_count": len(projected), "output_dir": output_dir, "fragments": manifest}
        print(json.dumps(payload, indent=2))
        return
    else:
        if migration_pipeline is not None:
            stitched = migration_pipeline.run(audio, args.transcript, legacy_severity)
        else:
            slice_runtime = build_stitching_slice(config)
            alignment = slice_runtime.aligner.align(audio, args.transcript)
            graph = slice_runtime.feature_extractor.build_phone_graph(alignment)
            graph = slice_runtime.complexity_scorer.score(graph)
            plan = slice_runtime.planner.plan(graph, slice_runtime.severity_profile)
            timing = slice_runtime.budget_manager.build(graph, plan)
            edited = slice_runtime.source_editor.apply(audio, graph, plan, timing)
            fragments = slice_runtime.fragment_synthesizer.synthesize(audio, graph, plan, timing)
            projected = slice_runtime.timbre_projector.project(audio, fragments)
            stitched = slice_runtime.assembler.assemble(audio, edited, projected)
        output_path = args.output or "stitched_output.wav"
        WavAudioWriter().write(output_path, stitched)
        payload = {
            "output_path": output_path,
            "sample_rate_hz": stitched.sample_rate_hz,
            "sample_count": len(stitched.samples),
            "metadata": stitched.metadata,
            "fragment_count": int(stitched.metadata.get("fragment_mix_count", "0")),
        }
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(asdict(payload), indent=2))


if __name__ == "__main__":
    main()
