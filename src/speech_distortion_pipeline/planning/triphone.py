from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from speech_distortion_pipeline.models import AudioBuffer
from speech_distortion_pipeline.models import (
    CandidateScore,
    PronunciationSkeleton,
    TriphoneEdge,
    TriphoneFeatureVector,
    TriphoneState,
    TraversalPath,
)
from speech_distortion_pipeline.phonology.phone_distance_reference import (
    ranked_candidate_entries_from_reference,
)
from speech_distortion_pipeline.resynthesis.phoneme_render_backend import (
    DEFAULT_KOKORO_VOICE,
    PHONEME_RENDER_BACKEND_KOKORO,
    PhonemeRenderError,
    render_with_phoneme_backend,
)


BOUNDARY_START = "<s>"
BOUNDARY_END = "</s>"


@dataclass(frozen=True)
class TriphoneReferenceEntry:
    state: TriphoneState
    embedding: tuple[float, ...]
    distance_to_source: float
    phonotactic_weight: float


@dataclass
class HeuristicTriphoneTraversalEngine:
    evidence_computer: object
    reference_bank_size: int = 20
    retrieval_top_k: int = 5
    bank_voice: str = DEFAULT_KOKORO_VOICE
    cache_root: str = "/data/raqchia/audio-assets/.cache"
    hubert_device: str = "cpu"
    _hubert_model: object | None = field(default=None, init=False, repr=False)
    _hubert_sample_rate: int = field(default=16000, init=False, repr=False)
    _hubert_backend_status: str = field(default="uninitialized", init=False, repr=False)
    _waveform_cache: dict[tuple[str, ...], np.ndarray | None] = field(default_factory=dict, init=False, repr=False)
    _embedding_cache: dict[tuple[str, ...], tuple[float, ...]] = field(default_factory=dict, init=False, repr=False)
    _reference_cache: dict[tuple[tuple[str, str, str], int, int], list[TriphoneReferenceEntry]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _class_bank_cache: dict[tuple[str, int, int], list[TriphoneState]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ensure_cache_dirs()

    def build_states(self, skeleton: PronunciationSkeleton) -> list[TriphoneState]:
        phones = list(skeleton.phone_sequence)
        states: list[TriphoneState] = []
        for index, phone in enumerate(phones):
            left = phones[index - 1] if index > 0 else BOUNDARY_START
            right = phones[index + 1] if index < len(phones) - 1 else BOUNDARY_END
            stress = "primary" if skeleton.primary_vowel_index == skeleton.phone_indices[index] else None
            if index == 0:
                syllable_role = "onset"
            elif index == len(phones) - 1:
                syllable_role = "coda"
            elif self._is_vowel(phone):
                syllable_role = "nucleus"
            else:
                syllable_role = "medial"
            states.append(
                TriphoneState(
                    left=left,
                    center=phone,
                    right=right,
                    left_boundary=(left == BOUNDARY_START),
                    right_boundary=(right == BOUNDARY_END),
                    stress=stress,
                    syllable_role=syllable_role,
                    word_position=index,
                )
            )
        return states

    def extract_triphone_features(self, state: TriphoneState) -> TriphoneFeatureVector:
        center = self._feature_bundle(state.center)
        left = self._feature_bundle(state.left)
        right = self._feature_bundle(state.right)
        left_distance = self._feature_distance(state.center, state.left)
        right_distance = self._feature_distance(state.center, state.right)
        left_right_compatibility = max(0.0, 1.0 - ((left_distance + right_distance) / 2.0))
        sonority_transition = self._sonority_transition_score(state.left, state.center, state.right)
        cluster_legality = self._cluster_legality_score(state.left, state.center, state.right)
        vowel_harmony = self._vowel_harmony_score(state.left, state.center, state.right)
        syllable_role_score = self._syllable_role_score(state)
        phonotactic_likelihood = max(
            0.0,
            min(
                1.0,
                (left_right_compatibility * 0.24)
                + (sonority_transition * 0.22)
                + (cluster_legality * 0.22)
                + (vowel_harmony * 0.16)
                + (syllable_role_score * 0.16),
            ),
        )
        return TriphoneFeatureVector(
            center_features={
                "place": center.get("place"),
                "manner": center.get("manner"),
                "voicing": center.get("voicing"),
                "vowel_height": center.get("vowel_height"),
                "vowel_backness": center.get("vowel_backness"),
                "rounding": center.get("rounding"),
                "sonority": center.get("sonority", 0.0),
            },
            left_right_compatibility=round(left_right_compatibility, 4),
            sonority_transition=round(sonority_transition, 4),
            consonant_cluster_legality=round(cluster_legality, 4),
            vowel_harmony=round(vowel_harmony, 4),
            syllable_role_score=round(syllable_role_score, 4),
            phonotactic_likelihood=round(phonotactic_likelihood, 4),
        )

    def is_valid_triphone(self, state: TriphoneState) -> bool:
        return self.phonotactic_weight(state) >= 0.08

    def phonotactic_weight(self, state: TriphoneState) -> float:
        features = self.extract_triphone_features(state)
        penalty = 0.0
        if state.left == state.center == state.right and not self._is_vowel(state.center):
            penalty += 0.45
        if self._is_consonant(state.left) and self._is_consonant(state.center) and self._is_consonant(state.right):
            penalty += 0.18
        if state.left == BOUNDARY_START and self._is_vowel(state.center) and self._is_vowel(state.right):
            penalty += 0.12
        return round(max(0.0, min(1.0, features.phonotactic_likelihood - penalty)), 4)

    def contextual_transition_cost(
        self,
        source: TriphoneState,
        target: TriphoneState,
        hubert_weight: float,
    ) -> TriphoneEdge:
        source_features = self.extract_triphone_features(source)
        target_features = self.extract_triphone_features(target)
        articulatory_distance = self._feature_distance(source.center, target.center)
        coarticulatory_disruption = max(
            0.0,
            source_features.left_right_compatibility - target_features.left_right_compatibility,
        )
        phonotactic_penalty = max(0.0, 1.0 - self.phonotactic_weight(target))
        sonority_violation = max(0.0, source_features.sonority_transition - target_features.sonority_transition)
        syllable_instability = abs(source_features.syllable_role_score - target_features.syllable_role_score)
        inertia = self.perceptual_inertia_cost(source.center, target.center)
        edge_cost = (
            (articulatory_distance * 0.28)
            + (coarticulatory_disruption * 0.18)
            + (phonotactic_penalty * 0.18)
            + (sonority_violation * 0.12)
            + (syllable_instability * 0.10)
            + (inertia * 0.14)
            + (hubert_weight * 0.18)
        )
        return TriphoneEdge(
            source=source,
            target=target,
            edge_cost=round(edge_cost, 4),
            feature_similarity=round(max(0.0, 1.0 - articulatory_distance), 4),
            phonotactic_likelihood=round(self.phonotactic_weight(target), 4),
            hubert_weight=round(hubert_weight, 4),
            perceptual_inertia=round(inertia, 4),
            cost_breakdown={
                "articulatory_distance": round(articulatory_distance, 4),
                "coarticulatory_disruption": round(coarticulatory_disruption, 4),
                "phonotactic_penalty": round(phonotactic_penalty, 4),
                "sonority_violation": round(sonority_violation, 4),
                "syllable_instability": round(syllable_instability, 4),
                "perceptual_inertia": round(inertia, 4),
                "hubert_weight": round(hubert_weight, 4),
            },
        )

    def perceptual_inertia_cost(self, left_phone: str, right_phone: str) -> float:
        if left_phone == right_phone:
            return 0.0
        feature_distance = self._feature_distance(left_phone, right_phone)
        left = self._feature_bundle(left_phone)
        right = self._feature_bundle(right_phone)
        penalty = feature_distance * 0.65
        if left.get("manner") != right.get("manner"):
            penalty += 0.15
        if self._is_vowel(left_phone) != self._is_vowel(right_phone):
            penalty += 0.18
        if left.get("sonority", 0.0) and right.get("sonority", 0.0):
            penalty += min(0.12, abs(float(left["sonority"]) - float(right["sonority"])) * 0.04)
        return round(max(0.0, min(1.0, penalty)), 4)

    def naturalness_score(self, phones: Sequence[str]) -> float:
        if not phones:
            return 0.0
        states = self._states_from_phones(list(phones))
        weights = [self.phonotactic_weight(state) for state in states]
        sonority = [self.extract_triphone_features(state).sonority_transition for state in states]
        vowels = sum(1 for phone in phones if self._is_vowel(phone))
        syllable_balance = max(0.0, min(1.0, vowels / float(max(1, len(phones) // 2))))
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (self._mean(weights, 0.0) * 0.45)
                    + (self._mean(sonority, 0.0) * 0.30)
                    + (syllable_balance * 0.25),
                ),
            ),
            4,
        )

    def generate_paths(
        self,
        skeleton: PronunciationSkeleton,
        jibberish_value: float,
        acoustic_features: dict[str, Any] | None = None,
        max_paths: int = 8,
    ) -> list[TraversalPath]:
        symbolic = self._beam_search(
            skeleton=skeleton,
            mode="symbolic_first",
            jibberish_value=jibberish_value,
            acoustic_features=acoustic_features or {},
            beam_width=max(3, min(10, max_paths)),
        )
        hubert = self._beam_search(
            skeleton=skeleton,
            mode="hubert_first",
            jibberish_value=jibberish_value,
            acoustic_features=acoustic_features or {},
            beam_width=max(3, min(10, max_paths)),
        )
        ordered = symbolic + hubert
        ordered.sort(
            key=lambda item: (
                item.total_cost,
                -item.naturalness_score,
                -item.continuity_score,
                item.grapheme,
            )
        )
        deduped: list[TraversalPath] = []
        seen = set()
        for path in ordered:
            key = (tuple(path.phones), path.grapheme)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped[:max_paths]

    def candidate_scores_from_paths(
        self,
        skeleton: PronunciationSkeleton,
        paths: Sequence[TraversalPath],
        duration_shape_similarity: float,
        similarity_threshold: float,
        max_safe_phoneme_distance: float,
    ) -> list[CandidateScore]:
        candidates: list[CandidateScore] = []
        for path in paths:
            phones = list(path.phones)
            grapheme = path.grapheme or self.surface_from_phones(phones)
            phoneme_distance = self._sequence_distance(skeleton.phone_sequence, phones, skeleton)
            similarity = self._pronunciation_similarity(skeleton, phones, duration_shape_similarity)
            candidates.append(
                CandidateScore(
                    word=skeleton.word,
                    word_index=skeleton.word_index,
                    grapheme=grapheme,
                    phones=phones,
                    grapheme_distance=self._grapheme_distance(skeleton.word, grapheme),
                    phoneme_distance=phoneme_distance,
                    pronunciation_similarity=similarity,
                    accepted=(similarity >= similarity_threshold and phoneme_distance <= max_safe_phoneme_distance),
                    rejection_reason="",
                    traversal_source=path.mode,
                    cost_breakdown=dict(path.feature_breakdown),
                )
            )
        for candidate in candidates:
            if not candidate.accepted:
                candidate.rejection_reason = (
                    "unsafe_phoneme_distance"
                    if candidate.phoneme_distance > max_safe_phoneme_distance
                    else "below_similarity_threshold"
                )
        return candidates

    def surface_from_phones(self, phones: Sequence[str]) -> str:
        mapping = {
            "P": "p",
            "B": "b",
            "T": "t",
            "D": "d",
            "K": "k",
            "G": "g",
            "M": "m",
            "N": "n",
            "NG": "ng",
            "F": "f",
            "V": "v",
            "S": "s",
            "Z": "z",
            "SH": "sh",
            "TH": "th",
            "DH": "dh",
            "CH": "ch",
            "JH": "j",
            "L": "l",
            "R": "r",
            "W": "w",
            "Y": "y",
            "HH": "h",
            "AE": "a",
            "AH": "uh",
            "EH": "e",
            "IH": "i",
            "IY": "ee",
            "OW": "o",
            "UW": "oo",
            "AY": "ai",
            "EY": "ay",
            "OY": "oy",
            "AW": "ow",
            "ER": "er",
            "AA": "ah",
        }
        return "".join(mapping.get(phone, phone.lower()) for phone in phones) or "uh"

    def _beam_search(
        self,
        skeleton: PronunciationSkeleton,
        mode: str,
        jibberish_value: float,
        acoustic_features: dict[str, Any],
        beam_width: int,
    ) -> list[TraversalPath]:
        original_states = self.build_states(skeleton)
        beam: list[tuple[list[TriphoneState], list[TriphoneEdge], float, list[str]]] = [([], [], 0.0, [])]
        for index, state in enumerate(original_states):
            candidates = self._proposals_for_state(
                state=state,
                mode=mode,
                jibberish_value=jibberish_value,
                acoustic_features=acoustic_features,
            )
            next_beam: list[tuple[list[TriphoneState], list[TriphoneEdge], float, list[str]]] = []
            for states, edges, total_cost, neighbors in beam:
                for candidate_state in candidates:
                    hubert_weight = self._hubert_transition_weight(
                        source=state,
                        target=candidate_state,
                        acoustic_features=acoustic_features,
                        mode=mode,
                    )
                    edge = self.contextual_transition_cost(state, candidate_state, hubert_weight)
                    next_states = states + [candidate_state]
                    next_edges = edges + [edge]
                    next_neighbors = neighbors + [self._state_signature(candidate_state)]
                    continuity_bias = self._continuity_bias(next_states)
                    diversity_weight = self._diversity_weight(next_states)
                    mode_bias = 0.02 if mode == "hubert_first" else 0.04
                    mutation_reward = 0.0
                    if candidate_state.center != state.center:
                        mutation_reward += 0.12 + (jibberish_value * (0.18 if mode == "hubert_first" else 0.10))
                    if candidate_state.left != state.left or candidate_state.right != state.right:
                        mutation_reward += 0.04 + (jibberish_value * (0.08 if mode == "hubert_first" else 0.03))
                    candidate_cost = (
                        total_cost
                        + edge.edge_cost
                        - continuity_bias
                        - diversity_weight
                        - mutation_reward
                        + mode_bias
                    )
                    next_beam.append((next_states, next_edges, candidate_cost, next_neighbors))
            next_beam.sort(key=lambda item: item[2])
            beam = next_beam[: max(beam_width, 1)]

        paths: list[TraversalPath] = []
        for states, edges, total_cost, neighbors in beam:
            phones = self._reconcile_states(states, skeleton.phone_sequence)
            grapheme = self.surface_from_phones(phones)
            naturalness = self.naturalness_score(phones)
            continuity = max(0.0, 1.0 - self._mean([edge.perceptual_inertia for edge in edges], 0.0))
            paths.append(
                TraversalPath(
                    mode=mode,
                    proposal_source=("hubert_reference_bank" if mode == "hubert_first" else "symbolic_neighbors"),
                    states=states,
                    edges=edges,
                    phones=phones,
                    grapheme=grapheme,
                    total_cost=round(max(0.0, total_cost), 4),
                    naturalness_score=naturalness,
                    continuity_score=round(continuity, 4),
                    diversity_score=round(self._diversity_weight(states), 4),
                    reference_neighbors=neighbors,
                    feature_breakdown={
                        "contextual_distance": round(self._mean([edge.edge_cost for edge in edges], 0.0), 4),
                        "phonotactic_penalty": round(
                            self._mean([1.0 - edge.phonotactic_likelihood for edge in edges], 0.0),
                            4,
                        ),
                        "perceptual_shift": round(self._mean([edge.perceptual_inertia for edge in edges], 0.0), 4),
                        "hubert_weight": round(self._mean([edge.hubert_weight for edge in edges], 0.0), 4),
                        "instability": round(max(0.0, 1.0 - naturalness), 4),
                    },
                )
            )
        return paths

    def _proposals_for_state(
        self,
        state: TriphoneState,
        mode: str,
        jibberish_value: float,
        acoustic_features: dict[str, Any],
    ) -> list[TriphoneState]:
        if mode == "hubert_first":
            hubert_neighbors = self._hubert_retrieve_neighbors(
                state=state,
                jibberish_value=jibberish_value,
                acoustic_features=acoustic_features,
            )
            if hubert_neighbors:
                return hubert_neighbors
        return self._neighbors_for_state(state, jibberish_value)

    def _neighbors_for_state(self, state: TriphoneState, jibberish_value: float) -> list[TriphoneState]:
        center_candidates = self._neighbor_phone_options(state.center, max_options=(6 if jibberish_value >= 0.75 else 4))
        neighbors = [state]
        for center in center_candidates:
            candidate = TriphoneState(
                left=state.left,
                center=center,
                right=state.right,
                left_boundary=state.left_boundary,
                right_boundary=state.right_boundary,
                stress=state.stress,
                syllable_role=state.syllable_role,
                word_position=state.word_position,
            )
            if self.is_valid_triphone(candidate) or candidate.center != state.center:
                neighbors.append(candidate)
        if jibberish_value >= 0.7:
            for left in self._neighbor_phone_options(state.left, max_options=2):
                candidate = TriphoneState(
                    left=left,
                    center=state.center,
                    right=state.right,
                    left_boundary=state.left_boundary,
                    right_boundary=state.right_boundary,
                    stress=state.stress,
                    syllable_role=state.syllable_role,
                    word_position=state.word_position,
                )
                if self.is_valid_triphone(candidate):
                    neighbors.append(candidate)
            for right in self._neighbor_phone_options(state.right, max_options=2):
                candidate = TriphoneState(
                    left=state.left,
                    center=state.center,
                    right=right,
                    left_boundary=state.left_boundary,
                    right_boundary=state.right_boundary,
                    stress=state.stress,
                    syllable_role=state.syllable_role,
                    word_position=state.word_position,
                )
                if self.is_valid_triphone(candidate):
                    neighbors.append(candidate)
        deduped: list[TriphoneState] = []
        seen = set()
        for neighbor in neighbors:
            key = (neighbor.left, neighbor.center, neighbor.right)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(neighbor)
        return deduped

    def _hubert_retrieve_neighbors(
        self,
        state: TriphoneState,
        jibberish_value: float,
        acoustic_features: dict[str, Any],
    ) -> list[TriphoneState]:
        bank = self._reference_bank_for_state(state=state, jibberish_value=jibberish_value)
        if not bank:
            return []
        allowed = max(3, min(self.retrieval_top_k, int(round(4 + (jibberish_value * 8)))))
        spread = max(1, int(round(1 + (jibberish_value * 4))))
        start_rank = 0 if jibberish_value <= 0.25 else min(len(bank) - 1, int(round(jibberish_value * 2)))
        ordered_states: list[TriphoneState] = [state]
        seen = {(state.left, state.center, state.right)}
        hubert_floor = float(acoustic_features.get("acoustic_embedding_distance", 0.0) or 0.0)
        for rank in range(start_rank, min(len(bank), allowed + start_rank)):
            entry = bank[rank]
            if rank > 0 and rank % spread != 0 and entry.distance_to_source < max(0.10, hubert_floor * 0.30):
                continue
            key = (entry.state.left, entry.state.center, entry.state.right)
            if key in seen:
                continue
            seen.add(key)
            ordered_states.append(entry.state)
        return ordered_states[:allowed]

    def _reference_bank_for_state(
        self,
        state: TriphoneState,
        jibberish_value: float,
    ) -> list[TriphoneReferenceEntry]:
        cache_key = (
            (state.left, state.center, state.right),
            int(round(jibberish_value * 100.0)),
            int(self.reference_bank_size),
        )
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return cached

        source_embedding = self._embedding_for_triphone(state)
        if source_embedding is None:
            self._reference_cache[cache_key] = []
            return []

        candidates = self._merge_bank_candidates(
            state=state,
            jibberish_value=jibberish_value,
            dynamic_candidates=self._bank_candidate_states(state, jibberish_value),
            class_candidates=self._prebuilt_class_bank_for_state(state, jibberish_value),
        )
        entries: list[TriphoneReferenceEntry] = []
        for candidate in candidates:
            embedding = self._embedding_for_triphone(candidate)
            if embedding is None:
                continue
            entries.append(
                TriphoneReferenceEntry(
                    state=candidate,
                    embedding=embedding,
                    distance_to_source=self._cosine_distance(source_embedding, embedding),
                    phonotactic_weight=self.phonotactic_weight(candidate),
                )
            )
        entries.sort(
            key=lambda item: (
                item.distance_to_source,
                -item.phonotactic_weight,
                self._feature_distance(state.center, item.state.center),
                item.state.center,
            )
        )
        self._reference_cache[cache_key] = entries[: self.reference_bank_size]
        return self._reference_cache[cache_key]

    def _bank_candidate_states(self, state: TriphoneState, jibberish_value: float) -> list[TriphoneState]:
        center_budget = max(5, min(10, int(round(5 + (jibberish_value * 5)))))
        context_budget = max(2, min(3, int(round(2 + (jibberish_value * 2)))))
        left_options = [state.left] if state.left_boundary else self._neighbor_phone_options(state.left, context_budget)
        center_options = self._neighbor_phone_options(state.center, center_budget)
        right_options = [state.right] if state.right_boundary else self._neighbor_phone_options(state.right, context_budget)
        bank: list[TriphoneState] = [state]
        for center in center_options[1:]:
            bank.append(self._make_state(state, left=state.left, center=center, right=state.right))
        if jibberish_value >= 0.45:
            for left in left_options[1:]:
                bank.append(self._make_state(state, left=left, center=state.center, right=state.right))
            for right in right_options[1:]:
                bank.append(self._make_state(state, left=state.left, center=state.center, right=right))
        if jibberish_value >= 0.75:
            for left in left_options[1:4]:
                for center in center_options[1:7]:
                    bank.append(self._make_state(state, left=left, center=center, right=state.right))
            for center in center_options[1:7]:
                for right in right_options[1:4]:
                    bank.append(self._make_state(state, left=state.left, center=center, right=right))
        deduped: list[TriphoneState] = []
        seen = set()
        for candidate in bank:
            key = (candidate.left, candidate.center, candidate.right)
            if key in seen:
                continue
            seen.add(key)
            if not self.is_valid_triphone(candidate):
                continue
            deduped.append(candidate)
        deduped.sort(
            key=lambda item: (
                -(
                    (self._phonotactic_mutation_reward(state, item) * 0.35)
                    + (self.phonotactic_weight(item) * 0.25)
                    + (self._feature_distance(state.center, item.center) * 0.20)
                    + (abs(self._sonority_of(state.center) - self._sonority_of(item.center)) * 0.04)
                ),
                item.center,
            )
        )
        return deduped[: self.reference_bank_size]

    def _merge_bank_candidates(
        self,
        state: TriphoneState,
        jibberish_value: float,
        dynamic_candidates: Sequence[TriphoneState],
        class_candidates: Sequence[TriphoneState],
    ) -> list[TriphoneState]:
        merged: list[TriphoneState] = []
        seen = set()
        preferred = list(dynamic_candidates) + list(class_candidates)
        max_candidates = max(self.reference_bank_size, int(round(10 + (jibberish_value * 8))))
        for candidate in preferred:
            key = (candidate.left, candidate.center, candidate.right)
            if key in seen:
                continue
            seen.add(key)
            if not self.is_valid_triphone(candidate):
                continue
            merged.append(candidate)
            if len(merged) >= max_candidates:
                break
        return merged

    def _prebuilt_class_bank_for_state(self, state: TriphoneState, jibberish_value: float) -> list[TriphoneState]:
        class_key = self._phone_class(state.center)
        cache_key = (
            class_key,
            int(round(jibberish_value * 100.0)),
            int(self.reference_bank_size),
        )
        cached = self._class_bank_cache.get(cache_key)
        if cached is not None:
            return cached
        disk_hit = self._load_class_bank_from_disk(cache_key, template=state)
        if disk_hit is not None:
            self._class_bank_cache[cache_key] = disk_hit
            return disk_hit

        prototypes = self._class_bank_prototypes(class_key, state)
        center_budget = max(4, min(8, int(round(4 + (jibberish_value * 4)))))
        class_bank: list[TriphoneState] = []
        for prototype in prototypes:
            center_options = self._neighbor_phone_options(prototype.center, center_budget)
            for center in center_options[:center_budget]:
                class_bank.append(
                    TriphoneState(
                        left=prototype.left,
                        center=center,
                        right=prototype.right,
                        left_boundary=prototype.left_boundary,
                        right_boundary=prototype.right_boundary,
                        stress=state.stress,
                        syllable_role=state.syllable_role,
                        word_position=state.word_position,
                    )
                )
        deduped: list[TriphoneState] = []
        seen = set()
        for candidate in class_bank:
            key = (candidate.left, candidate.center, candidate.right)
            if key in seen:
                continue
            seen.add(key)
            if not self.is_valid_triphone(candidate):
                continue
            deduped.append(candidate)
            if len(deduped) >= self.reference_bank_size:
                break
        self._class_bank_cache[cache_key] = deduped
        self._save_class_bank_to_disk(cache_key, deduped)
        return deduped

    def _neighbor_phone_options(self, phone: str, max_options: int) -> list[str]:
        if phone in {BOUNDARY_START, BOUNDARY_END}:
            return [phone]
        entries = ranked_candidate_entries_from_reference(phone)
        if entries:
            return [phone] + [entry["phones"][0] for entry in entries if len(entry["phones"]) == 1][:max_options]
        fallback = ["S", "Z", "T", "D", "N", "M", "L", "R", "OW", "AH", "EH", "IY"]
        return [phone] + [candidate for candidate in fallback if candidate != phone][:max_options]

    def _hubert_transition_weight(
        self,
        source: TriphoneState,
        target: TriphoneState,
        acoustic_features: dict[str, Any],
        mode: str,
    ) -> float:
        acoustic_distance = float(acoustic_features.get("acoustic_embedding_distance", 0.0) or 0.0)
        formant_distance = float(acoustic_features.get("formant_trajectory_distance", 0.0) or 0.0)
        proxy_distance = (self._feature_distance(source.center, target.center) * 0.55) + (formant_distance * 0.20)
        if mode == "hubert_first":
            retrieved_distance = self._hubert_neighbor_distance(source, target)
            return round(min(1.0, (retrieved_distance * 0.60) + (proxy_distance * 0.25) + (acoustic_distance * 0.15)), 4)
        return round(min(1.0, (acoustic_distance * 0.20) + (proxy_distance * 0.65)), 4)

    def _continuity_bias(self, states: Sequence[TriphoneState]) -> float:
        if len(states) <= 1:
            return 0.0
        transitions = [
            self.perceptual_inertia_cost(states[index - 1].center, states[index].center)
            for index in range(1, len(states))
        ]
        return round(max(0.0, 0.12 - (self._mean(transitions, 0.0) * 0.10)), 4)

    def _diversity_weight(self, states: Sequence[TriphoneState]) -> float:
        if not states:
            return 0.0
        unique_centers = len({state.center for state in states})
        return round(min(0.10, unique_centers / float(max(len(states), 1)) * 0.08), 4)

    def _reconcile_states(self, states: Sequence[TriphoneState], original_phones: Sequence[str]) -> list[str]:
        if not states:
            return list(original_phones)
        phones = [state.center for state in states]
        if len(phones) != len(original_phones):
            return list(original_phones)
        return phones

    def _states_from_phones(self, phones: list[str]) -> list[TriphoneState]:
        temp = PronunciationSkeleton(
            word="",
            word_index=0,
            phone_sequence=phones,
            phone_indices=list(range(len(phones))),
            anchor_phone_indices=[],
            anchor_phones=[],
            primary_vowel_index=None,
            primary_vowel_phone=None,
        )
        return self.build_states(temp)

    def _sequence_distance(
        self,
        original_phones: Sequence[str],
        candidate_phones: Sequence[str],
        skeleton: PronunciationSkeleton,
    ) -> float:
        mismatches = abs(len(original_phones) - len(candidate_phones))
        for index in range(min(len(original_phones), len(candidate_phones))):
            if original_phones[index] != candidate_phones[index]:
                mismatches += float(self.evidence_computer.feature_aware_phone_distance(original_phones[index], candidate_phones[index]))
                phone_index = skeleton.phone_indices[index]
                if phone_index in skeleton.anchor_phone_indices:
                    mismatches += 0.75
                if phone_index == skeleton.primary_vowel_index:
                    mismatches += 0.75
        return round(min(1.0, mismatches / float(max(len(original_phones), 1) * 2.25)), 4)

    def _pronunciation_similarity(
        self,
        skeleton: PronunciationSkeleton,
        phones: Sequence[str],
        duration_shape_similarity: float,
    ) -> float:
        return float(
            self.evidence_computer.pronunciation_similarity(
                skeleton=skeleton,
                candidate_phones=list(phones),
                duration_shape_similarity=duration_shape_similarity,
            )
        )

    def _grapheme_distance(self, original: str, candidate: str) -> float:
        return float(self.evidence_computer.weighted_grapheme_distance(original, candidate))

    def _feature_bundle(self, phone: str) -> dict[str, Any]:
        if phone in {BOUNDARY_START, BOUNDARY_END}:
            return {"sonority": 0.0, "boundary": True}
        bundle = self.evidence_computer._feature_bundle(phone)  # type: ignore[attr-defined]
        return {
            "place": bundle.place,
            "manner": bundle.manner,
            "voicing": bundle.voicing,
            "vowel_height": bundle.vowel_height,
            "vowel_backness": bundle.vowel_backness,
            "rounding": bundle.rounding,
            "sonority": getattr(bundle, "sonority", None) or self._sonority_of(phone),
        }

    def _feature_distance(self, left_phone: str, right_phone: str) -> float:
        if left_phone in {BOUNDARY_START, BOUNDARY_END} or right_phone in {BOUNDARY_START, BOUNDARY_END}:
            return 0.15 if left_phone != right_phone else 0.0
        return float(self.evidence_computer.feature_aware_phone_distance(left_phone, right_phone))

    def _sonority_transition_score(self, left: str, center: str, right: str) -> float:
        left_sonority = self._sonority_of(left)
        center_sonority = self._sonority_of(center)
        right_sonority = self._sonority_of(right)
        rise = max(0.0, center_sonority - left_sonority)
        fall = max(0.0, center_sonority - right_sonority)
        baseline = 0.35 + min(0.65, (rise * 0.18) + (fall * 0.18))
        return round(max(0.0, min(1.0, baseline)), 4)

    def _cluster_legality_score(self, left: str, center: str, right: str) -> float:
        illegal_clusters = {
            ("NG", "T"),
            ("ZH", "R"),
            ("HH", "NG"),
            ("DH", "L"),
        }
        pairs = [(left, center), (center, right)]
        penalty = 0.0
        for pair in pairs:
            if pair in illegal_clusters:
                penalty += 0.55
            if self._is_consonant(pair[0]) and self._is_consonant(pair[1]):
                if self._sonority_of(pair[1]) < self._sonority_of(pair[0]) - 2.2:
                    penalty += 0.20
        return round(max(0.0, min(1.0, 1.0 - penalty)), 4)

    def _vowel_harmony_score(self, left: str, center: str, right: str) -> float:
        vowels = [phone for phone in (left, center, right) if self._is_vowel(phone)]
        if len(vowels) <= 1:
            return 0.7
        compatibilities = []
        for index in range(1, len(vowels)):
            compatibilities.append(1.0 - self._feature_distance(vowels[index - 1], vowels[index]))
        return round(max(0.0, min(1.0, self._mean(compatibilities, 0.7))), 4)

    def _syllable_role_score(self, state: TriphoneState) -> float:
        if state.syllable_role == "nucleus":
            return 0.95
        if state.syllable_role == "onset":
            return 0.82 if self._is_consonant(state.center) else 0.55
        if state.syllable_role == "coda":
            return 0.80 if self._is_consonant(state.center) else 0.58
        return 0.68

    def _sonority_of(self, phone: str) -> float:
        sonority = {
            BOUNDARY_START: 0.0,
            BOUNDARY_END: 0.0,
            "P": 1.0, "B": 1.0, "T": 1.0, "D": 1.0, "K": 1.0, "G": 1.0,
            "CH": 1.5, "JH": 1.5,
            "F": 2.0, "V": 2.0, "S": 2.0, "Z": 2.0, "SH": 2.0, "TH": 2.0, "DH": 2.0, "HH": 2.0,
            "M": 3.0, "N": 3.0, "NG": 3.0,
            "L": 4.0, "R": 4.0,
            "W": 4.5, "Y": 4.5,
            "AE": 5.0, "AH": 5.0, "EH": 5.0, "IH": 5.0, "IY": 5.0, "OW": 5.0, "UW": 5.0,
            "AY": 5.0, "EY": 5.0, "OY": 5.0, "AW": 5.0, "ER": 5.0, "AA": 5.0,
        }
        return float(sonority.get(phone, 2.5))

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER", "AA"}

    def _is_consonant(self, phone: str) -> bool:
        return phone not in {BOUNDARY_START, BOUNDARY_END} and not self._is_vowel(phone)

    def _mean(self, values: Iterable[float], default: float) -> float:
        values = list(values)
        if not values:
            return default
        return sum(values) / float(len(values))

    def _phonotactic_mutation_reward(self, source: TriphoneState, target: TriphoneState) -> float:
        center_shift = self._feature_distance(source.center, target.center)
        context_shift = self._feature_distance(source.left, target.left) + self._feature_distance(source.right, target.right)
        return max(0.0, min(1.0, center_shift + (context_shift * 0.25)))

    def _state_signature(self, state: TriphoneState) -> str:
        return "{0}|{1}|{2}".format(state.left, state.center, state.right)

    def _make_state(self, template: TriphoneState, left: str, center: str, right: str) -> TriphoneState:
        return TriphoneState(
            left=left,
            center=center,
            right=right,
            left_boundary=template.left_boundary,
            right_boundary=template.right_boundary,
            stress=template.stress,
            syllable_role=template.syllable_role,
            word_position=template.word_position,
        )

    def _hubert_neighbor_distance(self, source: TriphoneState, target: TriphoneState) -> float:
        source_embedding = self._embedding_for_triphone(source)
        target_embedding = self._embedding_for_triphone(target)
        if source_embedding is None or target_embedding is None:
            return self._feature_distance(source.center, target.center)
        return self._cosine_distance(source_embedding, target_embedding)

    def _embedding_for_triphone(self, state: TriphoneState) -> tuple[float, ...] | None:
        cache_key = (state.left, state.center, state.right)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        disk_hit = self._load_embedding_from_disk(cache_key)
        if disk_hit is not None:
            self._embedding_cache[cache_key] = disk_hit
            return disk_hit
        model = self._load_hubert_model()
        if model is None:
            return None
        waveform = self._waveform_for_triphone(state)
        if waveform is None or waveform.size == 0:
            return None
        try:
            import torch
            import torchaudio

            tensor = torch.tensor(waveform, dtype=torch.float32).reshape(1, -1)
            if self._hubert_sample_rate != 16000:
                tensor = torchaudio.functional.resample(tensor, self._hubert_sample_rate, 16000)
            target_device = torch.device(self.hubert_device)
            tensor = tensor.to(target_device)
            with torch.no_grad():
                features = model.extract_features(tensor)
            if isinstance(features, tuple):
                features = features[0]
            if isinstance(features, list):
                features = features[-1]
            pooled = features.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            return None
        embedding = tuple(float(value) for value in pooled.tolist())
        self._embedding_cache[cache_key] = embedding
        self._save_embedding_to_disk(cache_key, embedding)
        return embedding

    def _waveform_for_triphone(self, state: TriphoneState) -> np.ndarray | None:
        cache_key = (state.left, state.center, state.right)
        if cache_key in self._waveform_cache:
            return self._waveform_cache[cache_key]
        disk_hit = self._load_waveform_from_disk(cache_key)
        if disk_hit is not None:
            # `None` means "render failed previously" and is cached too.
            self._waveform_cache[cache_key] = disk_hit
            return disk_hit
        if self._waveform_on_disk(cache_key):
            self._delete_waveform_on_disk(cache_key)
        waveform = self._render_triphone_waveform(state)
        self._waveform_cache[cache_key] = waveform
        self._save_waveform_to_disk(cache_key, waveform)
        return waveform

    def _render_triphone_waveform(self, state: TriphoneState) -> np.ndarray | None:
        phones = [phone for phone in (state.left, state.center, state.right) if phone not in {BOUNDARY_START, BOUNDARY_END}]
        if not phones:
            return None
        try:
            rendered = render_with_phoneme_backend(
                audio=AudioBuffer(samples=[0.0], sample_rate_hz=16000),
                phones=phones,
                backend=PHONEME_RENDER_BACKEND_KOKORO,
                backend_voice=self.bank_voice,
            )
        except PhonemeRenderError:
            return None
        return np.asarray(rendered.samples, dtype=np.float32).reshape(-1)

    def _load_hubert_model(self) -> object | None:
        if self._hubert_model is not None:
            return self._hubert_model
        if self._hubert_backend_status == "unavailable":
            return None
        try:
            import torch
            import torchaudio

            bundle = torchaudio.pipelines.HUBERT_BASE
            self._hubert_model = bundle.get_model()
            self._hubert_model = self._hubert_model.to(torch.device(self.hubert_device))
            self._hubert_model.eval()
            self._hubert_sample_rate = int(getattr(bundle, "sample_rate", 16000))
            self._hubert_backend_status = "torchaudio_pretrained"
            return self._hubert_model
        except Exception:
            self._hubert_backend_status = "unavailable"
            return None

    def hubert_backend_status(self) -> str:
        return self._hubert_backend_status

    def _cosine_distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        left_vec = np.asarray(left, dtype=np.float32)
        right_vec = np.asarray(right, dtype=np.float32)
        if left_vec.size == 0 or right_vec.size == 0:
            return 1.0
        left_norm = float(np.linalg.norm(left_vec))
        right_norm = float(np.linalg.norm(right_vec))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return 1.0
        similarity = float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
        return round(max(0.0, min(1.0, 1.0 - similarity)), 4)

    def _phone_class(self, phone: str) -> str:
        if phone in {BOUNDARY_START, BOUNDARY_END}:
            return "boundary"
        bundle = self._feature_bundle(phone)
        manner = str(bundle.get("manner") or "")
        if self._is_vowel(phone):
            height = str(bundle.get("vowel_height") or "mid")
            backness = str(bundle.get("vowel_backness") or "central")
            return "vowel_{0}_{1}".format(height, backness)
        place = str(bundle.get("place") or "other")
        return "{0}_{1}".format(manner or "consonant", place)

    def _class_bank_prototypes(self, class_key: str, state: TriphoneState) -> list[TriphoneState]:
        if class_key.startswith("vowel_"):
            center_family = [state.center, "IY", "EH", "AH", "OW", "UW", "AA"]
            left_contexts = ["Y", "W", "T", "S", "B"]
            right_contexts = ["N", "T", "L", "R", BOUNDARY_END]
        elif class_key.startswith("stop_"):
            center_family = [state.center, "T", "D", "K", "G", "P", "B"]
            left_contexts = [BOUNDARY_START, "S", "R", "N", "L"]
            right_contexts = ["R", "L", "AH", "OW", BOUNDARY_END]
        elif class_key.startswith("fricative_"):
            center_family = [state.center, "S", "Z", "SH", "F", "V", "TH"]
            left_contexts = [BOUNDARY_START, "R", "N", "T", "L"]
            right_contexts = ["T", "R", "AH", "OW", BOUNDARY_END]
        elif class_key.startswith("nasal_"):
            center_family = [state.center, "N", "M", "NG", "L", "R"]
            left_contexts = [BOUNDARY_START, "S", "T", "AH", "OW"]
            right_contexts = ["T", "D", "AH", "OW", BOUNDARY_END]
        elif class_key.startswith("liquid_") or class_key.startswith("glide_"):
            center_family = [state.center, "L", "R", "W", "Y", "N"]
            left_contexts = [BOUNDARY_START, "B", "T", "S", "K"]
            right_contexts = ["AH", "OW", "IY", "N", BOUNDARY_END]
        else:
            center_family = [state.center, "T", "S", "N", "AH", "OW"]
            left_contexts = [BOUNDARY_START, "S", "T", "N", "R"]
            right_contexts = ["AH", "OW", "N", "T", BOUNDARY_END]

        prototypes: list[TriphoneState] = []
        for left in left_contexts[:3]:
            for center in center_family[:5]:
                for right in right_contexts[:3]:
                    prototypes.append(
                        TriphoneState(
                            left=left,
                            center=center,
                            right=right,
                            left_boundary=(left == BOUNDARY_START),
                            right_boundary=(right == BOUNDARY_END),
                            stress=state.stress,
                            syllable_role=state.syllable_role,
                            word_position=state.word_position,
                        )
                    )
        prototypes.append(state)
        return prototypes

    def _ensure_cache_dirs(self) -> None:
        # Put all traversal caches under a stable user-provided cache root so
        # the expensive Kokoro renders and HuBERT embeddings survive restarts.
        base = Path(self.cache_root).expanduser()
        root = base / "speech-distortion" / "triphone_bank_v1"
        root.mkdir(parents=True, exist_ok=True)
        (root / "waveforms").mkdir(parents=True, exist_ok=True)
        (root / "embeddings").mkdir(parents=True, exist_ok=True)
        (root / "class_banks").mkdir(parents=True, exist_ok=True)

    def _cache_dir(self) -> Path:
        return Path(self.cache_root).expanduser() / "speech-distortion" / "triphone_bank_v1"

    def _safe_key(self, items: Sequence[str]) -> str:
        raw = "__".join(str(item) for item in items)
        return "".join(ch if (ch.isalnum() or ch in {"_", "-"} ) else "_" for ch in raw)[:160]

    def _waveform_path(self, key: tuple[str, ...]) -> Path:
        return self._cache_dir() / "waveforms" / f"{self._safe_key(key)}__{self._safe_key([self.bank_voice])}.npy"

    def _embedding_path(self, key: tuple[str, ...]) -> Path:
        return self._cache_dir() / "embeddings" / f"{self._safe_key(key)}__{self._safe_key([self.bank_voice])}.npy"

    def _class_bank_path(self, key: tuple[str, int, int]) -> Path:
        class_key, jitter_bucket, bank_size = key
        name = self._safe_key([class_key, str(jitter_bucket), str(bank_size), self.bank_voice])
        return self._cache_dir() / "class_banks" / f"{name}.json"

    def _waveform_on_disk(self, key: tuple[str, ...]) -> bool:
        return self._waveform_path(key).exists()

    def _save_waveform_to_disk(self, key: tuple[str, ...], waveform: np.ndarray | None) -> None:
        if waveform is None:
            return
        path = self._waveform_path(key)
        try:
            np.save(path, np.asarray(waveform, dtype=np.float32).reshape(-1))
        except Exception:
            return

    def _load_waveform_from_disk(self, key: tuple[str, ...]) -> np.ndarray | None:
        path = self._waveform_path(key)
        if not path.exists():
            return None
        try:
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        return arr

    def _delete_waveform_on_disk(self, key: tuple[str, ...]) -> None:
        path = self._waveform_path(key)
        try:
            if path.exists():
                path.unlink()
        except Exception:
            return

    def _save_embedding_to_disk(self, key: tuple[str, ...], embedding: tuple[float, ...]) -> None:
        path = self._embedding_path(key)
        try:
            np.save(path, np.asarray(list(embedding), dtype=np.float32))
        except Exception:
            return

    def _load_embedding_from_disk(self, key: tuple[str, ...]) -> tuple[float, ...] | None:
        path = self._embedding_path(key)
        if not path.exists():
            return None
        try:
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        return tuple(float(v) for v in arr.tolist())

    def _save_class_bank_to_disk(self, key: tuple[str, int, int], states: Sequence[TriphoneState]) -> None:
        path = self._class_bank_path(key)
        payload = [
            {
                "left": s.left,
                "center": s.center,
                "right": s.right,
                "left_boundary": bool(s.left_boundary),
                "right_boundary": bool(s.right_boundary),
            }
            for s in states
        ]
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            return

    def _load_class_bank_from_disk(
        self,
        key: tuple[str, int, int],
        template: TriphoneState,
    ) -> list[TriphoneState] | None:
        path = self._class_bank_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        out: list[TriphoneState] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            out.append(
                TriphoneState(
                    left=str(item.get("left", BOUNDARY_START)),
                    center=str(item.get("center", "")),
                    right=str(item.get("right", BOUNDARY_END)),
                    left_boundary=bool(item.get("left_boundary", False)),
                    right_boundary=bool(item.get("right_boundary", False)),
                    stress=template.stress,
                    syllable_role=template.syllable_role,
                    word_position=template.word_position,
                )
            )
        return out
