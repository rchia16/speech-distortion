#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_distortion_pipeline.bootstrap import build_hybrid_planning_slice, build_slider_controls
from speech_distortion_pipeline.config import load_hybrid_config
from speech_distortion_pipeline.io import WavAudioReader


DEFAULT_WORDS: tuple[str, ...] = (
    "no",
    "yes",
    "stop",
    "go",
    "pain",
    "help",
    "bath",
    "food",
    # "string",
    # "bath",
    # "think",
    # "this",
    # "judge",
    # "measure",
    # "river",
    # "yellow",
    # "around",
    # "school",
    # "street",
    # "glove",
    # "world",
    # "music",
    # "quick",
    # "zebra",
    # "orange",
    # "future",
    # "anchor",
    # "vowel",
    # "rhythm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-warm pronunciation_safe triphone/Kokoro/HuBERT caches on disk."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hybrid.yaml"),
        help="Path to hybrid config YAML.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=("male", "female", "man", "woman", "m", "f"),
        help="Reference gender used for alignment/planning warmup.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/raqchia/audio-assets/.cache"),
        help="Root cache dir expected by pronunciation-safe triphone cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda:0", "cuda:1"),
        help="Device for HuBERT embedding prewarm.",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=[],
        help="Extra words to warm in addition to defaults + acceptance-test words.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Optional cap on number of warmup words (0 means no cap).",
    )
    parser.add_argument(
        "--skip-acceptance-words",
        action="store_true",
        help="Do not include words from hybrid.yaml acceptance_tests.",
    )
    parser.add_argument(
        "--skip-direct-embedding-prewarm",
        action="store_true",
        help="Skip explicit triphone embedding prewarm and rely only on planner traversal.",
    )
    parser.add_argument(
        "--fail-if-no-embedding",
        action="store_true",
        help="Return non-zero exit if no embeddings are computed in this run.",
    )
    return parser.parse_args()


def dedupe_words(words: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for word in words:
        normalized = str(word).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def build_word_list(args: argparse.Namespace, config_words: Iterable[str]) -> list[str]:
    words = list(DEFAULT_WORDS)
    if not args.skip_acceptance_words:
        words.extend(config_words)
    words.extend(args.words or [])
    final = dedupe_words(words)
    if args.max_words and args.max_words > 0:
        return final[: args.max_words]
    return final


def summarize_cache(cache_root: Path) -> dict[str, int]:
    traversal_root = cache_root / Path("speech-distortion") /\
            Path("triphone_bank_v1")
    return {
        "waveforms": len(list((traversal_root / "waveforms").glob("*.npy"))),
        "embeddings": len(list((traversal_root / "embeddings").glob("*.npy"))),
        "class_banks": len(list((traversal_root / "class_banks").glob("*.json"))),
    }


def main() -> int:
    args = parse_args()
    gender = (args.gender or "female").strip().lower()
    config = load_hybrid_config(args.config)
    planning = build_hybrid_planning_slice(config)
    if gender in {"male", "m", "man"}:
        cache_partition = "man"
        prefix = "aus-en_man"
        root = "/data/raqchia/audio-assets/en.text-to-speech.online/aus-en_man"
    else:
        cache_partition = "woman"
        prefix = "aus-en_woman"
        root = "/data/raqchia/audio-assets/en.text-to-speech.online/aus-en_woman"
    cache_root = str(args.cache_root / cache_partition)

    acceptance_words = [case.word for case in config.acceptance_tests if case.word]
    words = build_word_list(args, acceptance_words)

    triphone_engine = planning.planner._triphone_engine  # warm cache explicitly for pronunciation-safe path
    triphone_engine.cache_root = cache_root
    triphone_engine.hubert_device = args.device
    triphone_engine.__post_init__()
    # Force early model initialization so failures are visible up front.
    _ = triphone_engine._load_hubert_model()

    slider_levels = (0.25, 0.55, 0.85, 1.0)
    print(f"[prewarm] cache_root={cache_root}")
    print(f"[prewarm] cache_partition={cache_partition}")
    print(f"[prewarm] device={args.device}")
    print(f"[prewarm] words={len(words)} levels={slider_levels}")
    print(f"[prewarm] hubert_backend_status_init={triphone_engine.hubert_backend_status()}")
    embedding_count_before = summarize_cache(Path(cache_root))["embeddings"]

    warmed = 0
    embedding_attempts = 0
    embedding_success = 0
    embedding_fail = 0

    for word in words:
        audio_file = Path(root) / Path(prefix+f"_{word}.wav")
        print("working with", audio_file)
        audio = WavAudioReader().read(str(audio_file))
        try:
            alignment = planning.aligner.align(audio, word)
            graph = planning.feature_extractor.build_phone_graph(alignment)
            for level in slider_levels:
                controls = build_slider_controls(
                    config,
                    jibberish=level,
                    clarity=0.0,
                    timing_instability=0.0,
                    allow_full_gibberish=False,
                )
                plan = planning.planner.plan(graph, controls, audio=audio)
                # Ensure traversal generation runs for each level.
                _ = len(plan.traversal_paths)
                if not args.skip_direct_embedding_prewarm:
                    for skeleton in plan.skeletons:
                        states = triphone_engine.build_states(skeleton)
                        for state in states:
                            # Directly materialize state embeddings on disk.
                            embedding_attempts += 1
                            embedding = triphone_engine._embedding_for_triphone(state)
                            if embedding is None:
                                embedding_fail += 1
                            else:
                                embedding_success += 1
                            # Materialize retrieval-bank embeddings for this jibberish level.
                            _ = triphone_engine._reference_bank_for_state(state, level)
            warmed += 1
            print(f"[prewarm] ok {word}")
        except Exception as exc:  # pragma: no cover - best-effort warmup utility
            print(f"[prewarm] skip {word}: {exc}")

    counts = summarize_cache(cache_root)
    embedding_count_after = counts["embeddings"]
    print(f"[prewarm] completed={warmed}/{len(words)}")
    print(
        "[prewarm] cache_counts waveforms={waveforms} embeddings={embeddings} class_banks={class_banks}".format(
            **counts
        )
    )
    print(f"[prewarm] embeddings_added={max(0, embedding_count_after - embedding_count_before)}")
    print(
        f"[prewarm] embedding_attempts={embedding_attempts} success={embedding_success} fail={embedding_fail}"
    )
    print(f"[prewarm] hubert_backend_status={triphone_engine.hubert_backend_status()}")
    if args.fail_if_no_embedding and embedding_success <= 0:
        print("[prewarm] error: no embeddings were computed in this run")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
