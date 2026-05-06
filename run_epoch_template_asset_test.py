#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from select_blabber_asset import load_asset_index, normalize_voice_mode, select_asset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/Users/160843/Downloads/UTS_for Ray/dataset_cache/Daniel")


def _serialize_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _serialize_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def load_template_module(template_version: str) -> Any:
    module_name = "Template_l2_compare_v2" if template_version == "v2" else "Template_l2_compare"
    return importlib.import_module(module_name)


def configure_template_module(module: Any, dataset_root: Path) -> None:
    module.BASE_ROOT = str(dataset_root)
    if hasattr(module, "patch_pickle_classes"):
        module.patch_pickle_classes()


def load_dataset(module: Any, day: int, session: int) -> Any:
    session_path = Path(module.path_builder(day, session))
    return module.read_pkl(str(session_path))


def resolve_label_id(dataset: Any, label_id: int | None, word: str | None) -> tuple[int, str]:
    words = list(dataset.final_id2word)
    if label_id is not None:
        if label_id < 0 or label_id >= len(words):
            raise ValueError(f"label_id={label_id} is outside available range 0..{len(words) - 1}.")
        return int(label_id), str(words[label_id])

    if word:
        normalized_word = word.strip().lower()
        for index, candidate in enumerate(words):
            if str(candidate).strip().lower() == normalized_word:
                return index, str(candidate)
        raise ValueError(f"Word '{word}' was not found in dataset labels: {', '.join(map(str, words))}")

    raise ValueError("Either --label-id or --word is required.")


def build_template(
    module: Any,
    template_day: int,
    template_sessions: list[int],
) -> dict[str, Any]:
    return module.build_template_from_dataset(
        template_days=[template_day],
        template_sessions=template_sessions,
    )


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    asset_index_path = Path(args.asset_index).expanduser().resolve()
    template_module = load_template_module(args.template_version)
    configure_template_module(template_module, dataset_root)

    eval_dataset = load_dataset(template_module, args.eval_day, args.eval_session)
    resolved_label_id, resolved_word = resolve_label_id(eval_dataset, args.label_id, args.word)
    trial, label_name, _dataset = template_module.get_one_trial_from_dataset(
        args.eval_day,
        args.eval_session,
        resolved_label_id,
        trial_index_within_label=args.trial_index,
    )
    template_object = build_template(
        template_module,
        args.template_day,
        [int(value) for value in args.template_sessions],
    )
    comparison_result = template_module.compare_signal_to_prebuilt_template(
        template_object=template_object,
        input_signal=trial,
        input_label=resolved_label_id,
        output_mode="both",
        sr=int(getattr(eval_dataset, "final_sampling_rate", args.sample_rate)),
    )

    candidates = load_asset_index(asset_index_path)
    selected = select_asset(
        comparison={
            "times": _serialize_json_safe(comparison_result["times"]),
            "per_time_l2": _serialize_json_safe(comparison_result["per_time_l2"]),
            "label_name": str(label_name),
            "label_id": resolved_label_id,
            "dtw_cost": _serialize_json_safe(comparison_result.get("dtw_cost")),
            "source": "Template_l2_compare_v2" if args.template_version == "v2" else "Template_l2_compare",
            "raw": _serialize_json_safe(comparison_result),
        },
        candidates=candidates,
        requested_voice=normalize_voice_mode(args.voice),
        target_word=resolved_word,
        target_label_id=resolved_label_id if args.match_by_label_id else None,
    )

    return {
        "dataset_root": str(dataset_root),
        "template_version": args.template_version,
        "template_day": args.template_day,
        "template_sessions": list(args.template_sessions),
        "eval_day": args.eval_day,
        "eval_session": args.eval_session,
        "trial_index": args.trial_index,
        "label_id": resolved_label_id,
        "word": resolved_word,
        "asset_index": str(asset_index_path),
        "comparison_result": _serialize_json_safe(comparison_result),
        "selected_asset": selected,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load one EEG epoch from the Daniel dataset cache, compare it against a template, "
            "and select a matching Blabber asset from a local asset library."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Root of the Daniel dataset cache containing Day*/sess*.pkl files.",
    )
    parser.add_argument(
        "--asset-index",
        required=True,
        help="Path to the local JSON asset index consumed by select_blabber_asset.py.",
    )
    parser.add_argument(
        "--template-version",
        choices=("v1", "v2"),
        default="v2",
        help="Use Template_l2_compare.py or Template_l2_compare_v2.py.",
    )
    parser.add_argument("--template-day", type=int, default=1, help="Day used to build the template.")
    parser.add_argument(
        "--template-sessions",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Sessions used to build the template. Default: 1 2 3 4 5 6 7",
    )
    parser.add_argument("--eval-day", type=int, default=1, help="Day containing the held-out epoch.")
    parser.add_argument("--eval-session", type=int, default=8, help="Session containing the held-out epoch.")
    parser.add_argument("--label-id", type=int, default=None, help="Label id to evaluate.")
    parser.add_argument("--word", default="", help="Word to evaluate. Used if --label-id is omitted.")
    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
        help="Which trial within the selected label to evaluate.",
    )
    parser.add_argument(
        "--voice",
        default="female",
        help="Requested asset voice bank: female/woman or male/man.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=250,
        help="Fallback EEG sample rate if the dataset object does not expose final_sampling_rate.",
    )
    parser.add_argument(
        "--match-by-label-id",
        action="store_true",
        help="Also allow asset selection to match index items by label_id when present.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the full comparison and asset-selection result JSON.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_test(args)
    output = json.dumps(result, indent=2)
    if args.output_json.strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
