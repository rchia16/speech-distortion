import json
from pathlib import Path

from speech_distortion_pipeline.phonology.phone_distance_reference import (
    build_global_blabber_ladder,
    build_phone_blabber_ladder,
    bucketed_phone_sequence,
    load_phone_distance_candidate_entries,
    load_phone_distance_candidates,
    load_phone_distance_rankings,
    ranked_candidate_entries_from_reference,
    ranked_candidates_from_reference,
    resolve_global_blabber_sequences,
    resolve_per_phoneme_blabber_sequences,
    resolve_phone_blabber_sequence,
)


def test_load_phone_distance_candidates_from_new_schema(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_neighbors_near_to_far": ["SH", "Z", "TH"],
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["Z"], "distance": 0.2, "rank": 1},
                {"phones": ["SH", "Z"], "distance": 0.3, "rank": 2},
                {"phones": ["TH"], "distance": 0.4, "rank": 3},
            ],
            "ranked_candidates_near_to_far": [["SH"], ["Z"], ["SH", "Z"], ["TH"]],
        }
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_phone_distance_rankings.cache_info().currsize >= 0
    assert load_phone_distance_candidates.cache_info().currsize >= 0
    assert load_phone_distance_candidate_entries.cache_info().currsize >= 0
    assert ranked_candidates_from_reference("S", str(reference_path)) == [
        ["SH"],
        ["Z"],
        ["SH", "Z"],
        ["TH"],
    ]
    assert ranked_candidate_entries_from_reference("S", str(reference_path))[:2] == [
        {"phones": ["SH"], "distance": 0.1, "rank": 0},
        {"phones": ["Z"], "distance": 0.2, "rank": 1},
    ]
    assert load_phone_distance_rankings(str(reference_path))["S"] == ["SH", "Z", "TH"]


def test_load_phone_distance_candidates_falls_back_to_legacy_schema(tmp_path: Path) -> None:
    payload = [{"phone": "S", "ranked_neighbors_near_to_far": ["SH", "Z", "TH"]}]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ranked_candidates_from_reference("S", str(reference_path)) == [["SH"], ["Z"], ["TH"]]


def test_bucketed_phone_sequence_uses_ranked_candidates_directly(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_neighbors_near_to_far": ["SH", "Z", "TH", "F"],
            "ranked_candidates_near_to_far": [["SH"], ["Z"], ["SH", "Z"], ["TH"], ["F", "TH"]],
        }
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    assert bucketed_phone_sequence("S", 1.0, reference_path=str(reference_path)) == ["S"]
    assert bucketed_phone_sequence("S", 0.9, reference_path=str(reference_path)) == ["SH"]
    assert bucketed_phone_sequence("S", 0.8, reference_path=str(reference_path)) == ["Z"]
    assert bucketed_phone_sequence("S", 0.7, reference_path=str(reference_path)) == ["SH", "Z"]
    assert bucketed_phone_sequence("S", 0.0, reference_path=str(reference_path)) == ["F", "TH"]


def test_global_blabber_ladder_uses_distinct_distance_sorted_word_states(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_neighbors_near_to_far": ["SH", "Z"],
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["Z"], "distance": 0.2, "rank": 1},
            ],
        },
        {
            "phone": "T",
            "ranked_neighbors_near_to_far": ["D", "K"],
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["D"], "distance": 0.05, "rank": 0},
                {"phones": ["K"], "distance": 0.3, "rank": 1},
            ],
        },
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    ladder = build_global_blabber_ladder(
        ["S", "T"],
        fallback_map={},
        reference_path=str(reference_path),
        preset_count=5,
    )

    assert [item["mutated_phones"] for item in ladder[:5]] == [
        ["S", "T"],
        ["S", "D"],
        ["SH", "T"],
        ["SH", "D"],
        ["Z", "T"],
    ]
    assert [item["total_distance"] for item in ladder[:5]] == [0.0, 0.05, 0.1, 0.15, 0.2]


def test_resolve_global_blabber_sequences_maps_quality_to_unique_preset_states(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_neighbors_near_to_far": ["SH", "Z"],
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["Z"], "distance": 0.2, "rank": 1},
            ],
        }
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    seq_a, preset_a, _expanded_a, distance_a = resolve_global_blabber_sequences(
        ["S"],
        0.9,
        reference_path=str(reference_path),
        preset_count=10,
    )
    seq_b, preset_b, _expanded_b, distance_b = resolve_global_blabber_sequences(
        ["S"],
        0.8,
        reference_path=str(reference_path),
        preset_count=10,
    )

    assert preset_a != preset_b
    assert seq_a != seq_b
    assert distance_a < distance_b


def test_build_phone_blabber_ladder_samples_distinct_distance_steps(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["Z"], "distance": 0.2, "rank": 1},
                {"phones": ["TH"], "distance": 0.3, "rank": 2},
            ],
        }
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    ladder = build_phone_blabber_ladder("S", reference_path=str(reference_path), preset_count=4)

    assert [entry["candidate_sequence"] for entry in ladder] == [["S"], ["SH"], ["Z"], ["TH"]]
    assert [entry["distance"] for entry in ladder] == [0.0, 0.1, 0.2, 0.3]


def test_build_phone_blabber_ladder_samples_direct_ranked_candidates_from_source_phone(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["F", "Z"], "distance": 0.2, "rank": 1},
                {"phones": ["Z"], "distance": 0.3, "rank": 2},
                {"phones": ["Z", "T"], "distance": 0.4, "rank": 3},
            ],
        }
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    ladder = build_phone_blabber_ladder("S", reference_path=str(reference_path), preset_count=5)

    assert [entry["candidate_sequence"] for entry in ladder] == [
        ["S"],
        ["SH"],
        ["F", "Z"],
        ["Z"],
        ["Z", "T"],
    ]


def test_resolve_phone_blabber_sequence_maps_quality_to_per_phone_step(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["Z"], "distance": 0.2, "rank": 1},
                {"phones": ["TH"], "distance": 0.3, "rank": 2},
            ],
        }
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    seq_a, preset_a, distance_a = resolve_phone_blabber_sequence(
        "S",
        0.66,
        reference_path=str(reference_path),
        preset_count=4,
    )
    seq_b, preset_b, distance_b = resolve_phone_blabber_sequence(
        "S",
        0.33,
        reference_path=str(reference_path),
        preset_count=4,
    )

    assert preset_a == 1
    assert preset_b == 2
    assert seq_a == ["SH"]
    assert seq_b == ["Z"]
    assert distance_a < distance_b


def test_resolve_per_phoneme_blabber_sequences_returns_per_phone_steps_and_distances(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["SH"], "distance": 0.1, "rank": 0},
                {"phones": ["Z"], "distance": 0.2, "rank": 1},
            ],
        },
        {
            "phone": "T",
            "ranked_candidate_entries_near_to_far": [
                {"phones": ["D"], "distance": 0.05, "rank": 0},
                {"phones": ["K"], "distance": 0.3, "rank": 1},
            ],
        },
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    sequences, preset_indices, distances, total_distance, expansion_used, numeric_entries_used = resolve_per_phoneme_blabber_sequences(
        ["S", "T"],
        [0.66, 0.33],
        reference_path=str(reference_path),
        preset_count=4,
    )

    assert sequences == [["SH"], ["K"]]
    assert preset_indices == [1, 2]
    assert distances == [0.1, 0.3]
    assert total_distance == 0.4
    assert expansion_used is False
    assert numeric_entries_used is True


def test_resolve_per_phoneme_blabber_sequences_marks_legacy_reference_fallback(tmp_path: Path) -> None:
    payload = [
        {
            "phone": "S",
            "ranked_candidates_near_to_far": [["SH"], ["Z"], ["SH", "Z"]],
        },
        {
            "phone": "T",
            "ranked_neighbors_near_to_far": ["D", "K"],
        },
    ]
    reference_path = tmp_path / "phoneme_distances.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    sequences, preset_indices, distances, total_distance, expansion_used, numeric_entries_used = resolve_per_phoneme_blabber_sequences(
        ["S", "T"],
        [0.66, 0.33],
        reference_path=str(reference_path),
        preset_count=4,
    )

    assert sequences == [["SH"], ["K"]]
    assert preset_indices == [1, 2]
    assert distances == [1.0, 2.0]
    assert total_distance == 3.0
    assert expansion_used is False
    assert numeric_entries_used is False
