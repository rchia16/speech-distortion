#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_blabber_library import (  # noqa: E402
    COQUI_ENV_NAME,
    ensure_coqui_backend,
    load_manifest,
    process_entry,
)
from speech_distortion_pipeline.io import WavAudioReader, WavAudioWriter  # noqa: E402
from speech_distortion_pipeline.resynthesis.phoneme_render_backend import (  # noqa: E402
    PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
    PHONEME_RENDER_BACKEND_FASTPITCH,
    PHONEME_RENDER_BACKEND_KOKORO,
    probe_phoneme_render_backend,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct phoneme render backends against the current Blabber generation path. "
            "This runs the same manifest through multiple backends and prints probe/render results as JSON."
        )
    )
    parser.add_argument("--manifest", required=True, help="Path to a .csv, .tsv, .json, or .jsonl manifest.")
    parser.add_argument(
        "--generation-mode",
        choices=("global_soft", "per_phoneme"),
        default="global_soft",
        help="Generation mode to use for each backend comparison.",
    )
    parser.add_argument(
        "--global-max-phoneme-distance",
        type=float,
        default=None,
        help="Optional quality-0 word-distance endpoint for global_soft generation.",
    )
    parser.add_argument(
        "--output-root",
        default="comparison-assets",
        help="Root directory where per-backend temporary outputs will be written.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_coqui_backend(COQUI_ENV_NAME)
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    if not entries:
        raise ValueError("Manifest did not contain any entries.")

    reader = WavAudioReader()
    writer = WavAudioWriter()
    backends = (
        PHONEME_RENDER_BACKEND_COQUI_TACOTRON2_DDC_PH,
        PHONEME_RENDER_BACKEND_FASTPITCH,
        PHONEME_RENDER_BACKEND_KOKORO,
    )

    results: list[dict[str, object]] = []
    for backend in backends:
        probe = probe_phoneme_render_backend(backend, COQUI_ENV_NAME)
        backend_result: dict[str, object] = {
            "backend": backend,
            "probe": {
                "ok": probe.ok,
                "status": probe.status,
                "detail": probe.detail,
                "model_name": probe.model_name,
            },
        }
        if not probe.ok:
            results.append(backend_result)
            continue

        backend_output_dir = output_root / backend
        backend_output_dir.mkdir(parents=True, exist_ok=True)
        generated: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        for entry in entries:
            try:
                generated.append(
                    process_entry(
                        entry,
                        backend_output_dir,
                        overwrite=True,
                        reader=reader,
                        writer=writer,
                        generation_mode=str(args.generation_mode),
                        global_max_phoneme_distance=(
                            max(0.0, float(args.global_max_phoneme_distance))
                            if args.global_max_phoneme_distance is not None
                            else None
                        ),
                        phoneme_render_backend=backend,
                    )
                )
            except Exception as exc:
                errors.append({"transcript": entry.transcript.strip(), "error": str(exc)})
        backend_result["generated"] = generated
        backend_result["errors"] = errors
        results.append(backend_result)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
