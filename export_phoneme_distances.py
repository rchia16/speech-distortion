#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_distortion_pipeline.phonology.acn_neighbors import (
    DEFAULT_PHONE_INVENTORY,
    phone_distance_rankings,
    phone_sequence_distance_entries,
    phone_sequence_distance_rankings,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ACN embedding-based phoneme distance rankings to a reference file."
    )
    parser.add_argument(
        "--output",
        default="phoneme_distances.json",
        help="Output path. Supported suffixes: .json or .csv",
    )
    parser.add_argument(
        "--top-n-pairs",
        type=int,
        default=8,
        help="Number of top single-phone neighbors used to form ordered 2-phone candidates.",
    )
    return parser


def export_json(
    output_path: Path,
    rankings: dict[str, list[str]],
    sequence_entries: dict[str, list[dict[str, object]]],
    sequence_rankings: dict[str, list[list[str]]],
) -> None:
    payload = []
    for phone in DEFAULT_PHONE_INVENTORY:
        neighbors = rankings.get(phone, [])
        candidates = sequence_rankings.get(phone, [])
        entries = sequence_entries.get(phone, [])
        payload.append(
            {
                "phone": phone,
                "ranked_neighbors_near_to_far": neighbors,
                "ranked_single_neighbors_near_to_far": neighbors,
                "ranked_candidates_near_to_far": candidates,
                "ranked_candidate_entries_near_to_far": entries,
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_csv(
    output_path: Path,
    rankings: dict[str, list[str]],
    sequence_entries: dict[str, list[dict[str, object]]],
    sequence_rankings: dict[str, list[list[str]]],
) -> None:
    max_neighbors = max((len(v) for v in rankings.values()), default=0)
    max_candidates = max((len(v) for v in sequence_rankings.values()), default=0)
    fieldnames = (
        ["phone"]
        + [f"neighbor_{index + 1}" for index in range(max_neighbors)]
        + [f"candidate_{index + 1}" for index in range(max_candidates)]
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for phone in DEFAULT_PHONE_INVENTORY:
            neighbors = rankings.get(phone, [])
            candidates = sequence_rankings.get(phone, [])
            row = {"phone": phone}
            for index, neighbor in enumerate(neighbors):
                row[f"neighbor_{index + 1}"] = neighbor
            for index, entry in enumerate(sequence_entries.get(phone, [])):
                phones = entry.get("phones", [])
                distance = entry.get("distance")
                if isinstance(phones, list):
                    label = " ".join(str(part) for part in phones)
                else:
                    label = str(phones)
                row[f"candidate_{index + 1}"] = f"{label}|{float(distance):.8f}"
            writer.writerow(row)


def main() -> int:
    args = build_arg_parser().parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rankings = phone_distance_rankings()
    sequence_entries = phone_sequence_distance_entries(top_n_pairs=max(0, args.top_n_pairs))
    sequence_rankings = phone_sequence_distance_rankings(top_n_pairs=max(0, args.top_n_pairs))
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        export_json(output_path, rankings, sequence_entries, sequence_rankings)
    elif suffix == ".csv":
        export_csv(output_path, rankings, sequence_entries, sequence_rankings)
    else:
        raise ValueError("Output path must end with .json or .csv")

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
