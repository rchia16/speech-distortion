from __future__ import annotations

import heapq
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_PATH = REPO_ROOT / "phoneme_distances.json"
GLOBAL_BLABBER_PRESET_COUNT = 20
PER_PHONEME_BLABBER_PRESET_COUNT = 20
GLOBAL_BLABBER_SCALED_LADDER_BEAM_WIDTH = 12


def _candidate_entry(
    phones: Sequence[str],
    distance: float,
    rank: int,
) -> dict[str, Any]:
    return {
        "phones": [part for part in phones if isinstance(part, str) and part],
        "distance": float(distance),
        "rank": int(rank),
    }


@lru_cache(maxsize=4)
def load_phone_distance_rankings(reference_path: str | None = None) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    payload = _load_reference_payload(reference_path)
    if not isinstance(payload, list):
        return rankings

    for item in payload:
        if not isinstance(item, dict):
            continue
        phone = item.get("phone")
        neighbors = item.get("ranked_neighbors_near_to_far")
        if not isinstance(phone, str) or not isinstance(neighbors, list):
            continue
        rankings[phone] = [neighbor for neighbor in neighbors if isinstance(neighbor, str)]
    return rankings


@lru_cache(maxsize=4)
def load_phone_distance_candidate_entries(reference_path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    payload = _load_reference_payload(reference_path)
    if not isinstance(payload, list):
        return rankings

    for item in payload:
        if not isinstance(item, dict):
            continue
        phone = item.get("phone")
        if not isinstance(phone, str):
            continue

        entries = _parse_candidate_entries(item)
        if entries:
            rankings[phone] = entries
    return rankings


@lru_cache(maxsize=4)
def load_phone_distance_entry_modes(reference_path: str | None = None) -> dict[str, str]:
    modes: dict[str, str] = {}
    payload = _load_reference_payload(reference_path)
    if not isinstance(payload, list):
        return modes

    for item in payload:
        if not isinstance(item, dict):
            continue
        phone = item.get("phone")
        if not isinstance(phone, str):
            continue
        if isinstance(item.get("ranked_candidate_entries_near_to_far"), list):
            modes[phone] = "numeric_entries"
        elif isinstance(item.get("ranked_candidates_near_to_far"), list):
            modes[phone] = "legacy_candidates"
        elif isinstance(item.get("ranked_neighbors_near_to_far"), list):
            modes[phone] = "legacy_neighbors"
    return modes


@lru_cache(maxsize=4)
def load_phone_distance_candidates(reference_path: str | None = None) -> dict[str, list[list[str]]]:
    return {
        phone: [list(entry["phones"]) for entry in entries]
        for phone, entries in load_phone_distance_candidate_entries(reference_path).items()
    }


def _parse_candidate_entries(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_entries = item.get("ranked_candidate_entries_near_to_far")
    if isinstance(raw_entries, list):
        parsed: list[dict[str, Any]] = []
        for index, entry in enumerate(raw_entries):
            normalized = _normalize_candidate_entry(entry, default_rank=index, default_distance=float(index + 1))
            if normalized is not None:
                parsed.append(normalized)
        if parsed:
            return _dedupe_candidate_entries(parsed)

    raw_candidates = item.get("ranked_candidates_near_to_far")
    if isinstance(raw_candidates, list):
        parsed = []
        for index, candidate in enumerate(raw_candidates):
            normalized = _normalize_candidate_entry(candidate, default_rank=index, default_distance=float(index + 1))
            if normalized is not None:
                parsed.append(normalized)
        if parsed:
            return _dedupe_candidate_entries(parsed)

    neighbors = item.get("ranked_neighbors_near_to_far")
    if isinstance(neighbors, list):
        parsed = []
        for index, neighbor in enumerate(neighbors):
            normalized = _normalize_candidate_entry(neighbor, default_rank=index, default_distance=float(index + 1))
            if normalized is not None:
                parsed.append(normalized)
        if parsed:
            return _dedupe_candidate_entries(parsed)
    return []


def _normalize_candidate_entry(
    entry: Any,
    default_rank: int,
    default_distance: float,
) -> dict[str, Any] | None:
    if isinstance(entry, str):
        phones = [entry]
        distance = default_distance
        rank = default_rank
    elif isinstance(entry, list):
        phones = [part for part in entry if isinstance(part, str) and part]
        distance = default_distance
        rank = default_rank
    elif isinstance(entry, dict):
        raw_phones = entry.get("phones")
        if isinstance(raw_phones, str):
            phones = [raw_phones]
        elif isinstance(raw_phones, list):
            phones = [part for part in raw_phones if isinstance(part, str) and part]
        else:
            return None
        raw_distance = entry.get("distance", default_distance)
        raw_rank = entry.get("rank", default_rank)
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError):
            distance = default_distance
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            rank = default_rank
    else:
        return None

    phones = [part for part in phones if part]
    if not phones:
        return None
    return _candidate_entry(phones, distance, rank)


def _load_reference_payload(reference_path: str | None = None) -> Any:
    path = Path(reference_path) if reference_path is not None else DEFAULT_REFERENCE_PATH
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ranked_neighbors_from_reference(
    phone: str,
    reference_path: str | None = None,
) -> list[str]:
    return list(load_phone_distance_rankings(reference_path).get(phone, ()))


def ranked_candidate_entries_from_reference(
    phone: str,
    reference_path: str | None = None,
) -> list[dict[str, Any]]:
    return [
        _candidate_entry(entry["phones"], entry["distance"], entry["rank"])
        for entry in load_phone_distance_candidate_entries(reference_path).get(phone, ())
    ]


def ranked_candidates_from_reference(
    phone: str,
    reference_path: str | None = None,
) -> list[list[str]]:
    return [list(entry["phones"]) for entry in ranked_candidate_entries_from_reference(phone, reference_path)]


def phone_distance_entry_mode(
    phone: str,
    reference_path: str | None = None,
) -> str:
    return load_phone_distance_entry_modes(reference_path).get(phone, "fallback_choices")


def phone_has_numeric_candidate_entries(
    phone: str,
    reference_path: str | None = None,
) -> bool:
    return phone_distance_entry_mode(phone, reference_path) == "numeric_entries"


def ranked_single_candidates_from_reference(
    phone: str,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
) -> list[list[str]]:
    return [
        list(entry["phones"])
        for entry in _effective_candidate_entries_pool(phone, fallback_choices, reference_path)
        if len(entry["phones"]) == 1
    ]


def ranked_multi_candidates_from_reference(
    phone: str,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
) -> list[list[str]]:
    return [
        list(entry["phones"])
        for entry in _effective_candidate_entries_pool(phone, fallback_choices, reference_path)
        if len(entry["phones"]) > 1
    ]


def top_ranked_neighbors_from_reference(
    phone: str,
    limit: int = 3,
    reference_path: str | None = None,
) -> list[str]:
    if limit <= 0:
        return []
    return ranked_neighbors_from_reference(phone, reference_path)[:limit]


def quality_grade_index(quality: float) -> int:
    clamped = max(0.0, min(1.0, quality))
    if clamped >= 0.999:
        return 0
    return min(10, max(0, int(round((1.0 - clamped) * 10.0))))


def global_blabber_preset_index(
    quality: float,
    preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
) -> int:
    clamped = max(0.0, min(1.0, quality))
    if preset_count <= 1 or clamped >= 0.999:
        return 0
    return min(preset_count - 1, max(1, int(round((1.0 - clamped) * (preset_count - 1)))))


def quality_for_global_blabber_preset(
    preset_index: int,
    preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
) -> float:
    if preset_count <= 1 or preset_index <= 0:
        return 1.0
    clamped_index = min(preset_count - 1, max(0, int(preset_index)))
    return max(0.0, min(1.0, 1.0 - (clamped_index / float(preset_count - 1))))


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    for item in items:
        if item and item not in ordered:
            ordered.append(item)
    return ordered


def _dedupe_candidate_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for entry in entries:
        phones = tuple(str(part) for part in entry.get("phones", ()) if isinstance(part, str) and part)
        if not phones or phones in seen:
            continue
        seen.add(phones)
        distance = float(entry.get("distance", len(deduped) + 1))
        rank = int(entry.get("rank", len(deduped)))
        deduped.append(_candidate_entry(phones, distance, rank))
    deduped.sort(key=lambda entry: _candidate_entry_sort_key(entry))
    return deduped


def _candidate_entry_sort_key(entry: Mapping[str, Any]) -> tuple[float, int, int, tuple[str, ...]]:
    phones = tuple(str(part) for part in entry.get("phones", ()) if isinstance(part, str) and part)
    distance = float(entry.get("distance", 0.0))
    rank = int(entry.get("rank", 0))
    return (distance, len(phones), rank, phones)


def _effective_candidate_entries_pool(
    phone: str,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
    max_phoneme_distance: float | None = None,
) -> list[dict[str, Any]]:
    ranked = ranked_candidate_entries_from_reference(phone, reference_path)
    if ranked:
        entries = _dedupe_candidate_entries(ranked)
    else:
        entries = [
            _candidate_entry([choice], float(index + 1), index)
            for index, choice in enumerate(_dedupe_preserve_order(fallback_choices))
        ]

    if max_phoneme_distance is None:
        return entries
    capped_distance = max(0.0, float(max_phoneme_distance))
    return [entry for entry in entries if float(entry["distance"]) <= capped_distance]


def _effective_candidate_pool(
    phone: str,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
) -> list[list[str]]:
    return [list(entry["phones"]) for entry in _effective_candidate_entries_pool(phone, fallback_choices, reference_path)]


def _candidate_sequence_for_grade(
    phone: str,
    grade_index: int,
    choices: Sequence[Sequence[str]],
) -> list[str]:
    if grade_index <= 0:
        return [phone]
    if not choices:
        return [phone]
    candidate_index = min(len(choices) - 1, grade_index - 1)
    sequence = [part for part in choices[candidate_index] if part]
    return sequence or [phone]


def _base_global_option(phone: str) -> dict[str, Any]:
    return _candidate_entry([phone], 0.0, -1)


def _global_options_for_phone(
    phone: str,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
    max_phoneme_distance: float | None = None,
) -> list[dict[str, Any]]:
    options = [_base_global_option(phone)]
    options.extend(
        _effective_candidate_entries_pool(
            phone,
            fallback_choices,
            reference_path,
            max_phoneme_distance=max_phoneme_distance,
        )
    )
    return _dedupe_candidate_entries(options)


def _global_state_key(
    indices: tuple[int, ...],
    options_by_phone: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[float, int, int, tuple[int, ...]]:
    total_distance = sum(float(options_by_phone[pos][choice]["distance"]) for pos, choice in enumerate(indices))
    expansion_count = sum(max(0, len(options_by_phone[pos][choice]["phones"]) - 1) for pos, choice in enumerate(indices))
    mutation_count = sum(1 for choice in indices if choice > 0)
    return (total_distance, expansion_count, mutation_count, indices)


def _global_ladder_entry_from_indices(
    indices: tuple[int, ...],
    options_by_phone: Sequence[Sequence[Mapping[str, Any]]],
    preset_index: int,
) -> dict[str, Any]:
    sequences = [list(options_by_phone[pos][choice]["phones"]) for pos, choice in enumerate(indices)]
    mutated = [phone for sequence in sequences for phone in sequence]
    total_distance = sum(float(options_by_phone[pos][choice]["distance"]) for pos, choice in enumerate(indices))
    return {
        "candidate_sequences": sequences,
        "mutated_phones": mutated,
        "total_distance": float(total_distance),
        "preset_index": int(preset_index),
        "expansion_used": any(len(sequence) > 1 for sequence in sequences),
    }


def _scaled_distance_targets(max_distance: float, preset_count: int) -> list[float]:
    target_count = max(1, int(preset_count))
    if target_count <= 1:
        return [0.0]
    return [
        float(max_distance) * (slot / float(target_count - 1))
        for slot in range(target_count)
    ]


def _state_target_sort_key(
    state: tuple[float, int, int, tuple[int, ...]],
    target_distance: float,
) -> tuple[float, bool, int, int, tuple[int, ...]]:
    total_distance, expansion_count, mutation_count, indices = state
    return (
        abs(total_distance - target_distance),
        total_distance > target_distance,
        expansion_count,
        mutation_count,
        indices,
    )


def _prune_index_states_to_targets(
    states: Sequence[tuple[float, int, int, tuple[int, ...]]],
    targets: Sequence[float],
    beam_width: int = GLOBAL_BLABBER_SCALED_LADDER_BEAM_WIDTH,
) -> list[tuple[float, int, int, tuple[int, ...]]]:
    selected: dict[tuple[int, ...], tuple[float, int, int, tuple[int, ...]]] = {}
    width = max(1, int(beam_width))
    for target in targets:
        ranked = sorted(states, key=lambda state, target=target: _state_target_sort_key(state, target))
        for state in ranked[:width]:
            selected[state[3]] = state
    return sorted(selected.values(), key=lambda state: (state[0], state[1], state[2], state[3]))


def _build_scaled_global_index_states(
    options_by_phone: Sequence[Sequence[Mapping[str, Any]]],
    preset_count: int,
    max_distance: float,
) -> list[tuple[float, int, int, tuple[int, ...]]]:
    target_count = max(1, int(preset_count))
    partial_states: list[tuple[float, int, int, tuple[int, ...]]] = [(0.0, 0, 0, tuple())]
    target_max_distance = max(0.0, float(max_distance))

    for options in options_by_phone:
        if not options:
            continue
        expanded_states: list[tuple[float, int, int, tuple[int, ...]]] = []
        for total_distance, expansion_count, mutation_count, indices in partial_states:
            for choice, option in enumerate(options):
                sequence = list(option["phones"])
                next_total_distance = total_distance + float(option["distance"])
                if next_total_distance > target_max_distance:
                    continue
                expanded_states.append(
                    (
                        next_total_distance,
                        expansion_count + max(0, len(sequence) - 1),
                        mutation_count + (1 if choice > 0 else 0),
                        indices + (choice,),
                    )
                )
        partial_targets = _scaled_distance_targets(target_max_distance, target_count)
        partial_states = _prune_index_states_to_targets(expanded_states, partial_targets)

    if not partial_states:
        return [(0.0, 0, 0, tuple(0 for _ in options_by_phone))]

    final_targets = _scaled_distance_targets(target_max_distance, target_count)
    return [
        min(partial_states, key=lambda state, target=target: _state_target_sort_key(state, target))
        for target in final_targets
    ]


def _sample_sorted_states(
    states: Sequence[Mapping[str, Any]],
    preset_count: int,
) -> list[dict[str, Any]]:
    if not states:
        return []
    target_count = max(1, int(preset_count))
    if len(states) <= target_count:
        sampled = [dict(state) for state in states]
        while len(sampled) < target_count:
            sampled.append(dict(sampled[-1]))
        return sampled

    positions: list[int] = []
    last_position = -1
    for slot in range(target_count):
        remaining_slots = target_count - slot - 1
        raw_position = int(round(slot * (len(states) - 1) / float(max(1, target_count - 1))))
        min_position = last_position + 1
        max_position = len(states) - remaining_slots - 1
        position = min(max(raw_position, min_position), max_position)
        positions.append(position)
        last_position = position
    return [dict(states[position]) for position in positions]


def build_phone_blabber_ladder(
    phone: str,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
    preset_count: int = PER_PHONEME_BLABBER_PRESET_COUNT,
) -> list[dict[str, Any]]:
    states = [_base_global_option(phone)]
    states.extend(_effective_candidate_entries_pool(phone, fallback_choices, reference_path))
    ladder = _sample_sorted_states(_dedupe_candidate_entries(states), preset_count)
    for index, entry in enumerate(ladder):
        entry["preset_index"] = index
        entry["candidate_sequence"] = list(entry["phones"])
        entry["expansion_used"] = len(entry["phones"]) > 1
    return ladder


def resolve_phone_blabber_sequence(
    phone: str,
    quality: float,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
    preset_count: int = PER_PHONEME_BLABBER_PRESET_COUNT,
) -> tuple[list[str], int, float]:
    preset_index = global_blabber_preset_index(quality, preset_count=preset_count)
    ladder = build_phone_blabber_ladder(
        phone,
        fallback_choices=fallback_choices,
        reference_path=reference_path,
        preset_count=preset_count,
    )
    selected = ladder[min(len(ladder) - 1, preset_index)]
    return list(selected["candidate_sequence"]), preset_index, float(selected["distance"])


def resolve_per_phoneme_blabber_sequences(
    source_phones: Sequence[str],
    phoneme_qualities: Sequence[float],
    fallback_map: Mapping[str, Sequence[str]] | None = None,
    reference_path: str | None = None,
    preset_count: int = PER_PHONEME_BLABBER_PRESET_COUNT,
) -> tuple[list[list[str]], list[int], list[float], float, bool, bool]:
    fallback_map = fallback_map or {}
    candidate_sequences: list[list[str]] = []
    preset_indices: list[int] = []
    distances: list[float] = []
    numeric_entries_used = True
    for index, phone in enumerate(source_phones):
        local_quality = float(phoneme_qualities[index]) if index < len(phoneme_qualities) else 1.0
        numeric_entries_used = numeric_entries_used and phone_has_numeric_candidate_entries(phone, reference_path)
        sequence, preset_index, distance = resolve_phone_blabber_sequence(
            phone,
            local_quality,
            fallback_choices=fallback_map.get(phone, ()),
            reference_path=reference_path,
            preset_count=preset_count,
        )
        candidate_sequences.append(sequence)
        preset_indices.append(preset_index)
        distances.append(distance)
    total_distance = float(sum(distances))
    expansion_used = any(len(sequence) > 1 for sequence in candidate_sequences)
    return candidate_sequences, preset_indices, distances, total_distance, \
            expansion_used, numeric_entries_used


def build_global_blabber_ladder(
    source_phones: Sequence[str],
    fallback_map: Mapping[str, Sequence[str]] | None = None,
    reference_path: str | None = None,
    preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
    max_phoneme_distance: float | None = None,
) -> list[dict[str, Any]]:
    if not source_phones:
        return []

    fallback_map = fallback_map or {}
    options_by_phone = [
        _global_options_for_phone(
            phone,
            fallback_map.get(phone, ()),
            reference_path,
            max_phoneme_distance=max_phoneme_distance,
        )
        for phone in source_phones
    ]
    if max_phoneme_distance is not None:
        scaled_states = _build_scaled_global_index_states(
            options_by_phone,
            preset_count,
            max_phoneme_distance,
        )
        return [
            _global_ladder_entry_from_indices(state[3], options_by_phone, preset_index)
            for preset_index, state in enumerate(scaled_states)
        ]

    start_indices = tuple(0 for _ in source_phones)
    heap: list[tuple[tuple[float, int, int, tuple[int, ...]], tuple[int, ...]]] = [
        (_global_state_key(start_indices, options_by_phone), start_indices)
    ]
    visited_index_states = {start_indices}
    ladder: list[dict[str, Any]] = []
    seen_mutated_sequences: set[tuple[str, ...]] = set()
    max_states = max(preset_count * 8, preset_count + 1, 32)

    while heap and len(ladder) < max_states:
        _key, indices = heapq.heappop(heap)
        entry = _global_ladder_entry_from_indices(indices, options_by_phone, len(ladder))
        mutated = tuple(entry["mutated_phones"])
        if mutated not in seen_mutated_sequences:
            seen_mutated_sequences.add(mutated)
            ladder.append(entry)

        for position in range(len(indices)):
            next_choice = indices[position] + 1
            if next_choice >= len(options_by_phone[position]):
                continue
            next_indices = list(indices)
            next_indices[position] = next_choice
            next_tuple = tuple(next_indices)
            if next_tuple in visited_index_states:
                continue
            visited_index_states.add(next_tuple)
            heapq.heappush(heap, (_global_state_key(next_tuple, options_by_phone), next_tuple))

    if not ladder:
        ladder.append(
            {
                "candidate_sequences": [[phone] for phone in source_phones],
                "mutated_phones": list(source_phones),
                "total_distance": 0.0,
                "preset_index": 0,
                "expansion_used": False,
            }
        )

    while len(ladder) < max(1, preset_count):
        previous = ladder[-1]
        ladder.append(
            {
                "candidate_sequences": [list(sequence) for sequence in previous["candidate_sequences"]],
                "mutated_phones": list(previous["mutated_phones"]),
                "total_distance": float(previous["total_distance"]),
                "preset_index": len(ladder),
                "expansion_used": bool(previous["expansion_used"]),
            }
        )

    return ladder[: max(1, preset_count)]


def resolve_global_blabber_sequences(
    source_phones: Sequence[str],
    quality: float,
    fallback_map: Mapping[str, Sequence[str]] | None = None,
    reference_path: str | None = None,
    preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
    max_phoneme_distance: float | None = None,
) -> tuple[list[list[str]], int, bool, float]:
    preset_index = global_blabber_preset_index(quality, preset_count=preset_count)
    if not source_phones:
        return [], preset_index, False, 0.0
    ladder = build_global_blabber_ladder(
        source_phones,
        fallback_map=fallback_map,
        reference_path=reference_path,
        preset_count=preset_count,
        max_phoneme_distance=max_phoneme_distance,
    )
    selected = ladder[min(len(ladder) - 1, preset_index)]
    return (
        [list(sequence) for sequence in selected["candidate_sequences"]],
        preset_index,
        bool(selected["expansion_used"]),
        float(selected["total_distance"]),
    )


def resolve_soft_global_blabber_sequences(
    source_phones: Sequence[str],
    quality: float,
    fallback_map: Mapping[str, Sequence[str]] | None = None,
    reference_path: str | None = None,
    preset_count: int = GLOBAL_BLABBER_PRESET_COUNT,
    max_phoneme_distance: float | None = None,
) -> tuple[list[list[str]], int, bool, float]:
    return resolve_global_blabber_sequences(
        source_phones,
        quality,
        fallback_map=fallback_map,
        reference_path=reference_path,
        preset_count=preset_count,
        max_phoneme_distance=max_phoneme_distance,
    )


def bucketed_phone_choice(
    phone: str,
    quality: float,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
) -> str:
    grade_index = quality_grade_index(quality)
    sequence = _candidate_sequence_for_grade(
        phone,
        grade_index,
        _effective_candidate_pool(phone, fallback_choices, reference_path),
    )
    return sequence[0]


def bucketed_phone_sequence(
    phone: str,
    quality: float,
    fallback_choices: Sequence[str] = (),
    reference_path: str | None = None,
) -> list[str]:
    return _candidate_sequence_for_grade(
        phone,
        quality_grade_index(quality),
        _effective_candidate_pool(phone, fallback_choices, reference_path),
    )
