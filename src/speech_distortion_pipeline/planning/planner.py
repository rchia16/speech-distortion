from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from speech_distortion_pipeline.config import HybridConfig
from speech_distortion_pipeline.models import (
    AudioBuffer,
    CandidateScore,
    ClarityPlan,
    DecisionTrace,
    DistanceEvidenceBundle,
    DistanceSignal,
    EditOperation,
    EditPlan,
    EditType,
    PhoneGraph,
    PhoneNode,
    PronunciationSafePlan,
    PronunciationSimilarityScore,
    PronunciationSkeleton,
    ProtectionMap,
    RepairAction,
    RoutingDecision,
    SeverityProfile,
    SimilarityComponents,
    SliderControls,
    SliderEstimate,
    WordComplexity,
)
from speech_distortion_pipeline.phonology import GraphemeToPhoneme
from .evidence import HeuristicDistanceEvidenceComputer
from .triphone import HeuristicTriphoneTraversalEngine

PHONE_THOLD = 1.0 # 0.55
MAX_CANDIDATES_0 = 12

class ErrorPlanner(Protocol):
    """Produces phone-level edit operations from phonological structure."""

    def plan(self, graph: PhoneGraph, severity: SeverityProfile) -> EditPlan:
        raise NotImplementedError


@dataclass
class HeuristicErrorPlanner:
    """Deterministic planner for the early prototype slices.

    The planner uses phone features, configured severity, and word complexity to
    emit a sparse set of edit operations that cover substitution, distorted
    substitution, addition, distortion, and local duration changes.
    """

    planner_name: str = "heuristic_error_planner_v1"

    _SIMPLE_SUBSTITUTIONS: Dict[str, List[str]] = None  # type: ignore[assignment]
    _DISTORTED_SUBSTITUTIONS: Dict[str, List[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._SIMPLE_SUBSTITUTIONS is None:
            self._SIMPLE_SUBSTITUTIONS = {
                "R": ["W"],
                "L": ["W"],
                "K": ["T"],
                "G": ["D"],
                "TH": ["F"],
                "DH": ["D"],
                "S": ["T"],
                "Z": ["D"],
            }
        if self._DISTORTED_SUBSTITUTIONS is None:
            self._DISTORTED_SUBSTITUTIONS = {
                "S": ["SH"],
                "Z": ["ZH"],
                "TH": ["S"],
                "R": ["R"],
                "L": ["L"],
                "P": ["P"],
                "T": ["T"],
                "K": ["K"],
            }

    def plan(self, graph: PhoneGraph, severity: SeverityProfile) -> EditPlan:
        operations: List[EditOperation] = []
        complexity_by_word = {item.index: item for item in graph.word_complexities}
        nodes_by_word: Dict[int, List[PhoneNode]] = {}
        for node in graph.nodes:
            nodes_by_word.setdefault(node.word_index, []).append(node)

        for word_index in sorted(nodes_by_word):
            word_nodes = nodes_by_word[word_index]
            complexity = complexity_by_word.get(word_index)
            weight = self._complexity_weight(complexity, severity)

            substitution = self._select_substitution(word_nodes, severity, weight)
            if substitution is not None:
                operations.append(substitution)

            distorted = self._select_distorted_substitution(word_nodes, severity, weight)
            if distorted is not None:
                operations.append(distorted)

            addition = self._select_addition(word_nodes, severity, weight)
            if addition is not None:
                operations.append(addition)

            distortion = self._select_distortion(word_nodes, severity, weight)
            if distortion is not None:
                operations.append(distortion)

            operations.extend(self._select_lengthening(word_nodes, severity, weight))

        operations.sort(key=lambda operation: (operation.target_phone_indices[0], operation.edit_type.value))
        return EditPlan(operations=operations, planner_name=self.planner_name)

    def _complexity_weight(
        self, complexity: Optional[WordComplexity], severity: SeverityProfile
    ) -> float:
        base = severity.global_severity
        if complexity is None:
            return round(base, 4)
        return round(base + (complexity.score * severity.complexity_slope), 4)

    def _select_substitution(
        self, nodes: List[PhoneNode], severity: SeverityProfile, weight: float
    ) -> Optional[EditOperation]:
        if severity.substitution_bias <= 0.0 or weight < 0.18:
            return None
        target = next((node for node in nodes if node.phone in self._SIMPLE_SUBSTITUTIONS), None)
        if target is None:
            return None
        return EditOperation(
            edit_type=EditType.SUBSTITUTE,
            target_phone_indices=[target.index],
            replacement_phones=list(self._SIMPLE_SUBSTITUTIONS[target.phone]),
            complexity_weight=weight,
            notes=[
                "simple substitution",
                "{0}->{1}".format(target.phone, "".join(self._SIMPLE_SUBSTITUTIONS[target.phone])),
            ],
        )

    def _select_distorted_substitution(
        self, nodes: List[PhoneNode], severity: SeverityProfile, weight: float
    ) -> Optional[EditOperation]:
        if severity.distortion_bias <= 0.0 or weight < 0.3:
            return None
        target = next(
            (
                node
                for node in nodes
                if node.phone in self._DISTORTED_SUBSTITUTIONS
                and (node.features.manner in {"fricative", "stop", "liquid"})
            ),
            None,
        )
        if target is None:
            return None

        notes = ["distorted substitution"]
        if target.phone == "S":
            notes.append("lateralized_or_slushy_s")
        elif target.phone == "P":
            notes.append("nasalized_stop_like_realization")
        else:
            notes.append("off_target_realization")

        return EditOperation(
            edit_type=EditType.DISTORTED_SUBSTITUTE,
            target_phone_indices=[target.index],
            replacement_phones=list(self._DISTORTED_SUBSTITUTIONS[target.phone]),
            complexity_weight=weight,
            notes=notes,
        )

    def _select_addition(
        self, nodes: List[PhoneNode], severity: SeverityProfile, weight: float
    ) -> Optional[EditOperation]:
        if severity.addition_bias <= 0.0 or weight < 0.25:
            return None
        cluster = [node for node in nodes if node.features.cluster_member]
        if len(cluster) < 2:
            return None
        inserted = ["AH"]
        return EditOperation(
            edit_type=EditType.ADD,
            target_phone_indices=[cluster[0].index],
            inserted_phones=inserted,
            complexity_weight=weight,
            notes=["epenthetic addition", "cluster_break"],
        )

    def _select_distortion(
        self, nodes: List[PhoneNode], severity: SeverityProfile, weight: float
    ) -> Optional[EditOperation]:
        if severity.distortion_bias <= 0.0 or weight < 0.22:
            return None
        target = next(
            (
                node
                for node in nodes
                if node.features.manner in {"fricative", "affricate", "stop"}
                or node.features.later_developing
            ),
            None,
        )
        if target is None:
            return None

        notes = ["segment distortion"]
        if target.phone in {"S", "Z", "SH"}:
            notes.append("spectral_slushiness")
        elif target.features.manner == "stop":
            notes.append("reduced_closure_precision")
        else:
            notes.append("off_target_segment")

        return EditOperation(
            edit_type=EditType.DISTORT,
            target_phone_indices=[target.index],
            complexity_weight=weight,
            notes=notes,
        )

    def _select_lengthening(
        self, nodes: List[PhoneNode], severity: SeverityProfile, weight: float
    ) -> List[EditOperation]:
        operations: List[EditOperation] = []
        vowel_target = next((node for node in nodes if self._is_vowel(node.phone)), None)
        if vowel_target is not None and severity.vowel_lengthening_bias > 0.0 and weight >= 0.15:
            operations.append(
                EditOperation(
                    edit_type=EditType.LENGTHEN_VOWEL,
                    target_phone_indices=[vowel_target.index],
                    target_duration_delta_sec=self._duration_delta(weight, severity.vowel_lengthening_bias, 0.035),
                    complexity_weight=weight,
                    notes=["local_vowel_lengthening"],
                )
            )

        consonant_target = next(
            (
                node
                for node in nodes
                if node.features.manner in {"fricative", "nasal", "stop"}
                and not self._is_vowel(node.phone)
            ),
            None,
        )
        if consonant_target is not None and severity.consonant_lengthening_bias > 0.0 and weight >= 0.2:
            operations.append(
                EditOperation(
                    edit_type=EditType.LENGTHEN_CONSONANT,
                    target_phone_indices=[consonant_target.index],
                    target_duration_delta_sec=self._duration_delta(
                        weight, severity.consonant_lengthening_bias, 0.025
                    ),
                    complexity_weight=weight,
                    notes=["local_consonant_lengthening"],
                )
            )

        return operations

    def _duration_delta(self, weight: float, bias: float, max_delta_sec: float) -> float:
        scaled = min(weight * max(bias, 0.0), 1.0)
        return round(max_delta_sec * scaled, 4)

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER"}


class HybridPlanner(Protocol):
    """Builds pronunciation-safe hybrid plans from slider controls and distance evidence."""

    def plan(
        self,
        graph: PhoneGraph,
        controls: SliderControls,
        audio: Optional[AudioBuffer] = None,
    ) -> PronunciationSafePlan:
        raise NotImplementedError


@dataclass
class HeuristicHybridPlanner:
    config: HybridConfig
    g2p: GraphemeToPhoneme
    planner_name: str = "heuristic_hybrid_planner_v1"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.global_constraints.random_seed)
        self._evidence_computer = HeuristicDistanceEvidenceComputer(config=self.config)
        self._triphone_engine = HeuristicTriphoneTraversalEngine(
            evidence_computer=self._evidence_computer
        )

    def plan(
        self,
        graph: PhoneGraph,
        controls: SliderControls,
        audio: Optional[AudioBuffer] = None,
    ) -> PronunciationSafePlan:
        skeletons = self._build_skeletons(graph)
        protection_maps = [self._build_protection_map(skeleton) for skeleton in skeletons]
        provisional_jibberish_estimate = SliderEstimate(
            value=controls.jibberish,
            confidence=0.85,
            evidence_sources=["control_prior"],
            routing_rule="control_prior",
        )

        candidate_seed_map: Dict[int, List[CandidateScore]] = {}
        for skeleton in skeletons:
            candidate_seed_map[skeleton.word_index] = self._build_candidate_scores(
                skeleton,
                provisional_jibberish_estimate,
                controls,
            )
        evidence_bundle = self._evidence_computer.compute(
            graph=graph,
            skeletons=skeletons,
            candidates_by_word=candidate_seed_map,
            controls=controls,
            audio=audio,
        )
        distance_evidence = evidence_bundle.signals
        slider_estimates = self._evidence_computer.estimate_sliders(evidence_bundle)

        candidates: List[CandidateScore] = []
        selected_candidates: List[CandidateScore] = []
        traversal_paths = []
        operations: List[EditOperation] = []
        for skeleton in skeletons:
            word_candidates = self._build_candidate_scores(
                skeleton,
                slider_estimates["jibberish"],
                controls,
                acoustic_features=distance_evidence.get(
                    "acoustic_embedding_distance",
                    DistanceSignal(name="acoustic_embedding_distance", value=0.0),
                ).raw_features,
            )
            candidates.extend(word_candidates)
            traversal_paths.extend(
                self._triphone_engine.generate_paths(
                    skeleton=skeleton,
                    jibberish_value=slider_estimates["jibberish"].value,
                    acoustic_features=distance_evidence.get(
                        "acoustic_embedding_distance",
                        DistanceSignal(name="acoustic_embedding_distance", value=0.0),
                    ).raw_features,
                    max_paths=8,
                )
            )
            selected = self._select_candidate(skeleton, word_candidates, slider_estimates["jibberish"])
            selected_candidates.append(selected)
            operations.extend(
                self._build_operations_for_candidate(
                    graph=graph,
                    skeleton=skeleton,
                    candidate=selected,
                    controls=controls,
                    jibberish_estimate=slider_estimates["jibberish"],
                )
            )

        clarity_plan = self._build_clarity_plan(slider_estimates["clarity"].value)
        generated_parameters = {
            "jibberish": self._derive_jibberish_parameters(slider_estimates["jibberish"].value, controls),
            "clarity": clarity_plan.parameters,
            "timing_instability": self._derive_timing_parameters(slider_estimates["timing_instability"].value),
        }

        repairs: List[RepairAction] = []
        similarity = self._score_plan(skeletons, operations, controls)
        if not similarity.passed:
            operations, repairs = self._repair_plan(skeletons, operations, controls)
            similarity = self._score_plan(skeletons, operations, controls)

        operations.sort(
            key=lambda item: (item.target_phone_indices[0], item.edit_type.value)
            if item.target_phone_indices
            else (10**9, item.edit_type.value)
        )
        return PronunciationSafePlan(
            controls=controls,
            skeletons=skeletons,
            protection_maps=protection_maps,
            operations=operations,
            clarity_plan=clarity_plan,
            similarity=similarity,
            repairs=repairs,
            generated_parameters=generated_parameters,
            slider_estimates=slider_estimates,
            distance_evidence=distance_evidence,
            evidence_bundle=evidence_bundle,
            candidates=candidates,
            selected_candidates=selected_candidates,
            traversal_paths=traversal_paths,
            trace=self._build_trace(
                skeletons=skeletons,
                operations=operations,
                selected_candidates=selected_candidates,
                distance_evidence=distance_evidence,
                slider_estimates=slider_estimates,
                repairs=repairs,
                similarity=similarity,
                candidates=candidates,
                evidence_bundle=evidence_bundle,
                traversal_paths=traversal_paths,
            ),
        )

    def _build_skeletons(self, graph: PhoneGraph) -> List[PronunciationSkeleton]:
        rules = self.config.pronunciation_skeleton.anchor_selection_rules
        by_word: Dict[int, List[PhoneNode]] = {}
        for node in graph.nodes:
            by_word.setdefault(node.word_index, []).append(node)

        skeletons: List[PronunciationSkeleton] = []
        for word_index in sorted(by_word):
            nodes = by_word[word_index]
            phone_sequence = [node.phone for node in nodes]
            phone_indices = [node.index for node in nodes]
            primary_vowel_node = next((node for node in nodes if self._is_vowel(node.phone)), None)
            stress_bearing = [node.index for node in nodes if node.stress is not None]
            if not stress_bearing and primary_vowel_node is not None:
                stress_bearing = [primary_vowel_node.index]
            syllable_count = max((node.syllable_index or 0) for node in nodes) + 1 if nodes else 1
            is_short_word = len(nodes) <= rules.short_word_anchor_all_phones_when_phone_count_lte

            anchor_indices: List[int] = []
            if is_short_word:
                anchor_indices = list(phone_indices)
            else:
                if rules.always_anchor_primary_vowel and primary_vowel_node is not None:
                    anchor_indices.append(primary_vowel_node.index)
                if rules.always_anchor_first_content_consonant:
                    first_consonant = next((node for node in nodes if not self._is_vowel(node.phone)), None)
                    if first_consonant is not None:
                        anchor_indices.append(first_consonant.index)
                if rules.anchor_final_consonant_for_closed_syllables:
                    final_consonant = next((node for node in reversed(nodes) if not self._is_vowel(node.phone)), None)
                    if final_consonant is not None:
                        anchor_indices.append(final_consonant.index)
            anchor_indices = sorted(set(anchor_indices))

            first_vowel_position = next((index for index, node in enumerate(nodes) if self._is_vowel(node.phone)), None)
            if first_vowel_position is None:
                onset_rime_shape = "consonant_only"
            else:
                onset = "".join("V" if self._is_vowel(node.phone) else "C" for node in nodes[:first_vowel_position])
                rime = "".join("V" if self._is_vowel(node.phone) else "C" for node in nodes[first_vowel_position:])
                onset_rime_shape = "{0}|{1}".format(onset or "-", rime or "-")

            start_shape = "vowel" if self._is_vowel(nodes[0].phone) else "consonant"
            end_shape = "vowel" if self._is_vowel(nodes[-1].phone) else "consonant"
            skeletons.append(
                PronunciationSkeleton(
                    word=nodes[0].word,
                    word_index=word_index,
                    phone_sequence=phone_sequence,
                    phone_indices=phone_indices,
                    anchor_phone_indices=anchor_indices,
                    anchor_phones=[node.phone for node in nodes if node.index in anchor_indices],
                    primary_vowel_index=primary_vowel_node.index if primary_vowel_node is not None else None,
                    primary_vowel_phone=primary_vowel_node.phone if primary_vowel_node is not None else None,
                    stress_bearing_phone_indices=stress_bearing,
                    syllable_count=syllable_count,
                    onset_rime_shape=onset_rime_shape,
                    word_boundary_shape="{0}_to_{1}".format(start_shape, end_shape),
                    manner_classes=[node.features.manner or "vowel" for node in nodes],
                    is_short_word=is_short_word,
                )
            )
        return skeletons

    def _build_protection_map(self, skeleton: PronunciationSkeleton) -> ProtectionMap:
        jibberish_guardrails = self.config.sliders["jibberish"].guardrails.values
        return ProtectionMap(
            protected_phone_indices=list(skeleton.anchor_phone_indices),
            anchor_phone_indices=list(skeleton.anchor_phone_indices),
            primary_vowel_index=skeleton.primary_vowel_index,
            preserve_phone_order=bool(jibberish_guardrails.get("preserve_phone_order", True)),
            preserve_syllable_count=bool(jibberish_guardrails.get("preserve_syllable_count", True)),
        )

    def _build_candidate_scores(
        self,
        skeleton: PronunciationSkeleton,
        jibberish_estimate: SliderEstimate,
        controls: SliderControls,
        acoustic_features: Optional[Dict[str, object]] = None,
    ) -> List[CandidateScore]:
        threshold = self._candidate_similarity_threshold(skeleton, controls)
        max_safe_phoneme_distance = 0.55
        if jibberish_estimate.value >= 0.85:
            max_safe_phoneme_distance = 0.68
        elif jibberish_estimate.value >= 0.65:
            max_safe_phoneme_distance = 0.62
        candidates: List[CandidateScore] = []
        duration_shape_similarity = max(0.0, 1.0 - min(1.0, controls.distance_overrides.get("duration_shape_distance", 0.0)))
        traversal_paths = self._triphone_engine.generate_paths(
            skeleton=skeleton,
            jibberish_value=jibberish_estimate.value,
            acoustic_features=dict(acoustic_features or {}),
            max_paths=max(MAX_CANDIDATES_0, int(round(6 + (jibberish_estimate.value * 10)))),
        )
        candidates.extend(
            self._triphone_engine.candidate_scores_from_paths(
                skeleton=skeleton,
                paths=traversal_paths,
                duration_shape_similarity=duration_shape_similarity,
                similarity_threshold=threshold,
                max_safe_phoneme_distance=max_safe_phoneme_distance,
            )
        )
        for variant in self._generate_grapheme_candidates(skeleton.word, jibberish_estimate.value):
            phones = self.g2p.phonemize(variant.replace("-", " "))
            phoneme_distance = self._phoneme_distance(skeleton.phone_sequence, phones, skeleton)
            pronunciation_similarity = self._candidate_similarity(skeleton, phones, duration_shape_similarity)
            accepted = pronunciation_similarity >= threshold and phoneme_distance <= max_safe_phoneme_distance
            rejection_reason = ""
            if not accepted:
                rejection_reason = "below_similarity_threshold"
                if phoneme_distance > max_safe_phoneme_distance:
                    rejection_reason = "unsafe_phoneme_distance"
            candidates.append(
                CandidateScore(
                    word=skeleton.word,
                    word_index=skeleton.word_index,
                    grapheme=variant,
                    phones=phones,
                    grapheme_distance=self._grapheme_distance(skeleton.word, variant),
                    phoneme_distance=phoneme_distance,
                    pronunciation_similarity=pronunciation_similarity,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    traversal_source="grapheme_fallback",
                )
            )
        deduped: Dict[tuple[str, tuple[str, ...]], CandidateScore] = {}
        for candidate in candidates:
            key = (candidate.grapheme, tuple(candidate.phones))
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = candidate
                continue
            if (
                candidate.accepted and not existing.accepted
            ) or (
                candidate.accepted == existing.accepted
                and (
                    candidate.pronunciation_similarity > existing.pronunciation_similarity
                    or candidate.phoneme_distance < existing.phoneme_distance
                )
            ):
                deduped[key] = candidate
        candidates = list(deduped.values())
        candidates.sort(
            key=lambda item: (
                not item.accepted,
                0 if item.traversal_source in {"symbolic_first", "hubert_first"} else 1,
                -(item.pronunciation_similarity),
                item.phoneme_distance,
                -item.grapheme_distance,
                item.grapheme,
            )
        )
        return candidates

    def _select_candidate(
        self,
            skeleton: PronunciationSkeleton,
            candidates: List[CandidateScore],
            jibberish_estimate: SliderEstimate,
    ) -> CandidateScore:
        accepted = [candidate for candidate in candidates if candidate.accepted]
        if not accepted:
            return CandidateScore(
                word=skeleton.word,
                word_index=skeleton.word_index,
                grapheme=skeleton.word,
                phones=list(skeleton.phone_sequence),
                pronunciation_similarity=1.0,
                accepted=True,
            )
        if jibberish_estimate.value <= 0.2:
            accepted.sort(
                key=lambda item: (
                    item.grapheme != skeleton.word,
                    -item.pronunciation_similarity,
                    item.phoneme_distance,
                    item.grapheme_distance,
                    item.grapheme,
                )
            )
            return accepted[0]

        if jibberish_estimate.value >= 0.6:
            contextual_accepted = [
                item for item in accepted if item.traversal_source in {"symbolic_first", "hubert_first"}
            ]
            if contextual_accepted:
                accepted = contextual_accepted

        if jibberish_estimate.value >= 0.75:
            strongly_mutated = [
                item
                for item in accepted
                if item.grapheme != skeleton.word
                and (
                    item.grapheme_distance >= 0.42
                    or abs(len(item.grapheme) - len(skeleton.word)) >= 2
                    or "-" in item.grapheme
                    or item.traversal_source in {"symbolic_first", "hubert_first"}
                )
            ]
            if strongly_mutated:
                accepted = strongly_mutated

        target_grapheme_distance = min(0.92, 0.24 + (jibberish_estimate.value * 0.58))
        target_visual_change = min(0.90, 0.20 + (jibberish_estimate.value * 0.65))
        accepted.sort(
            key=lambda item: (
                -(
                    (item.pronunciation_similarity * 0.34)
                    + ((1.0 - item.phoneme_distance) * 0.24)
                    + (
                        max(
                            0.0,
                            1.0 - abs(item.grapheme_distance - target_grapheme_distance) / max(target_grapheme_distance, 0.15),
                        )
                        * 0.22
                    )
                    + (
                        max(
                            0.0,
                            1.0 - abs(self._visual_change_score(skeleton.word, item.grapheme) - target_visual_change)
                            / max(target_visual_change, 0.18),
                        )
                        * 0.16
                    )
                    + ((1.0 if item.grapheme != skeleton.word else 0.0) * 0.12 * jibberish_estimate.value)
                    + (0.05 if item.traversal_source == "hubert_first" else 0.0)
                    + (0.03 if item.traversal_source == "symbolic_first" else 0.0)
                ),
                item.grapheme == skeleton.word and jibberish_estimate.value >= 0.45,
                item.grapheme,
            )
        )
        return accepted[0]

    def _visual_change_score(self, original: str, candidate: str) -> float:
        base = original.lower()
        variant = candidate.lower()
        score = self._grapheme_distance(base, variant) * 0.55
        if variant != base:
            score += 0.15
        if "-" in variant:
            score += 0.18
        score += min(0.2, abs(len(variant) - len(base)) * 0.07)
        if any(double in variant for double in ("aa", "ee", "ii", "oo", "uu")):
            score += 0.08
        if variant.endswith(("oh", "uh", "ah", "ayo", "owa", "iya")):
            score += 0.10
        return round(max(0.0, min(1.0, score)), 4)

    def _build_operations_for_candidate(
        self,
        graph: PhoneGraph,
        skeleton: PronunciationSkeleton,
        candidate: CandidateScore,
        controls: SliderControls,
        jibberish_estimate: SliderEstimate,
    ) -> List[EditOperation]:
        nodes_by_index = {node.index: node for node in graph.nodes}
        slider = self.config.sliders["jibberish"]
        guardrails = self._guardrails_for_skeleton(slider, skeleton)
        confidence_scale = max(0.4, jibberish_estimate.confidence)
        candidate_indices = [
            index
            for index in skeleton.phone_indices
            if index not in skeleton.anchor_phone_indices and index != skeleton.primary_vowel_index
        ]
        substitution_budget = int(
            min(
                len(candidate_indices),
                round(
                    len(skeleton.phone_indices)
                    * float(guardrails.get("max_phone_substitution_ratio", 0.0))
                    * confidence_scale
                ),
            )
        )
        operations: List[EditOperation] = []
        differing_positions = [
            position
            for position in range(min(len(skeleton.phone_sequence), len(candidate.phones)))
            if skeleton.phone_sequence[position] != candidate.phones[position]
        ]
        for position in differing_positions[:substitution_budget]:
            target_index = skeleton.phone_indices[position]
            if target_index in skeleton.anchor_phone_indices:
                continue
            node = nodes_by_index[target_index]
            operations.append(
                EditOperation(
                    edit_type=EditType.SUBSTITUTE,
                    target_phone_indices=[target_index],
                    replacement_phones=[candidate.phones[position]],
                    complexity_weight=round(jibberish_estimate.value, 4),
                    notes=[
                        "grapheme_guided_jibberish",
                        "candidate={0}".format(candidate.grapheme),
                        "distance={0}".format(int(round(1 + (candidate.phoneme_distance * 3)))),
                        "source_phone={0}".format(node.phone),
                    ],
                )
            )

        if not operations and candidate.grapheme != skeleton.word and jibberish_estimate.value >= 0.65:
            target_index = next(
                (
                    index
                    for index in skeleton.phone_indices
                    if index not in skeleton.anchor_phone_indices
                ),
                skeleton.phone_indices[0],
            )
            operations.append(
                EditOperation(
                    edit_type=EditType.DISTORT,
                    target_phone_indices=[target_index],
                    complexity_weight=round(jibberish_estimate.value, 4),
                    notes=["surface_variant_only", "candidate={0}".format(candidate.grapheme)],
                )
            )

        if not controls.allow_full_gibberish and jibberish_estimate.value >= 0.7 and skeleton.is_short_word:
            operations.append(
                EditOperation(
                    edit_type=EditType.DISTORT,
                    target_phone_indices=[skeleton.phone_indices[0]],
                    complexity_weight=round(jibberish_estimate.value, 4),
                    notes=["short_word_core_preserved"],
                )
            )
        return operations

    def _guardrails_for_skeleton(self, slider: object, skeleton: PronunciationSkeleton) -> Dict[str, object]:
        assert hasattr(slider, "guardrails")
        base = dict(slider.guardrails.values)
        short_word_mode = slider.guardrails.short_word_mode
        if skeleton.is_short_word and short_word_mode is not None:
            base["max_phone_substitution_ratio"] = short_word_mode.max_phone_substitution_ratio
            base["max_deletions_per_word"] = short_word_mode.max_deletions_per_word
            base["preserve_first_phone"] = short_word_mode.preserve_first_phone
            base["preserve_primary_vowel_nucleus"] = short_word_mode.preserve_primary_vowel_nucleus
            if short_word_mode.minimum_pronunciation_similarity is not None:
                base["minimum_pronunciation_similarity"] = short_word_mode.minimum_pronunciation_similarity
        return base

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER", "AA"}

    def _derive_jibberish_parameters(self, jibberish_value: float, controls: SliderControls) -> Dict[str, float]:
        guardrails = self.config.sliders["jibberish"].guardrails.values
        return {
            "substitution_probability": round(pow(jibberish_value, 1.2) * 0.60, 4),
            "substitution_distance": float(round(1 + (jibberish_value * 3))),
            "vowel_shift_probability": round(max(min((jibberish_value - 0.15) / 0.85, 1.0), 0.0) * 0.50, 4),
            "insertion_probability": round(max(min((jibberish_value - 0.45) / 0.55, 1.0), 0.0) * 0.30, 4),
            "deletion_probability": 0.0 if not controls.allow_full_gibberish else round(jibberish_value * 0.15, 4),
            "repetition_probability": round(max(min((jibberish_value - 0.60) / 0.40, 1.0), 0.0) * 0.25, 4),
            "max_anchor_phone_distance": float(guardrails.get("max_anchor_phone_distance", 1)),
        }

    def _derive_timing_parameters(self, timing_value: float) -> Dict[str, float]:
        guardrails = self.config.sliders["timing_instability"].guardrails.values
        max_pause = float(guardrails.get("max_pause_ms_inside_short_word", 180))
        return {
            "phone_duration_jitter": round(timing_value * 0.35, 4),
            "syllable_duration_jitter": round(timing_value * 0.25, 4),
            "pause_insertion_probability": round(max(min((timing_value - 0.30) / 0.70, 1.0), 0.0) * 0.35, 4),
            "pause_duration_ms": round(min(20 + (timing_value * 180), max_pause), 4),
            "micro_stutter_probability": round(max(min((timing_value - 0.45) / 0.55, 1.0), 0.0) * 0.45, 4),
            "onset_repeat_probability": round(max(min((timing_value - 0.55) / 0.45, 1.0), 0.0) * 0.35, 4),
        }

    def _build_clarity_plan(self, clarity_value: float) -> ClarityPlan:
        guardrails = dict(self.config.sliders["clarity"].guardrails.values)
        max_formant = float(guardrails.get("max_formant_centralization_for_anchor_vowels", 0.45))
        min_burst = float(guardrails.get("min_consonant_burst_energy_ratio", 0.35))
        max_blur_ms = float(guardrails.get("max_transition_blur_ms", 60))
        return ClarityPlan(
            parameters={
                "formant_centralization": round(min(clarity_value * 0.70, max_formant), 4),
                "spectral_tilt_db_per_octave": round(clarity_value * -5.0, 4),
                "high_frequency_reduction": round(clarity_value * 0.80, 4),
                "consonant_burst_softening": round(min(pow(clarity_value, 1.1) * 0.90, 1 - min_burst), 4),
                "fricative_smearing": round(clarity_value * 0.85, 4),
                "transition_blur_ms": round(min(5 + (clarity_value * 55), max_blur_ms), 4),
                "breath_noise_mix": round(clarity_value * 0.12, 4),
            },
            guardrails=guardrails,
        )

    def _score_plan(
        self, skeletons: List[PronunciationSkeleton], operations: List[EditOperation], controls: SliderControls
    ) -> PronunciationSimilarityScore:
        weights = self.config.pronunciation_similarity_score.weights
        op_by_index: Dict[int, EditOperation] = {}
        insertions = 0
        deletions = 0
        for operation in operations:
            if operation.target_phone_indices:
                op_by_index[operation.target_phone_indices[0]] = operation
            insertions += len(operation.inserted_phones)
            if not operation.replacement_phones and operation.edit_type == EditType.SUBSTITUTE:
                deletions += 1

        anchor_total = sum(len(skeleton.anchor_phone_indices) for skeleton in skeletons) or 1
        preserved_anchors = 0
        vowel_similarity_total = 0.0
        syllable_similarity_total = 0.0

        for skeleton in skeletons:
            for anchor_index in skeleton.anchor_phone_indices:
                operation = op_by_index.get(anchor_index)
                if operation is None or not operation.replacement_phones:
                    preserved_anchors += 1
                else:
                    original_phone = skeleton.phone_sequence[skeleton.phone_indices.index(anchor_index)]
                    if operation.replacement_phones[0] == original_phone:
                        preserved_anchors += 1

            if skeleton.primary_vowel_index is None:
                vowel_similarity_total += 1.0
            else:
                operation = op_by_index.get(skeleton.primary_vowel_index)
                if operation is None or not operation.replacement_phones:
                    vowel_similarity_total += 1.0
                elif self._is_vowel(operation.replacement_phones[0]):
                    vowel_similarity_total += 0.6

            syllable_penalty = min(1.0, (insertions + deletions) / max(len(skeleton.phone_indices), 1))
            syllable_similarity_total += max(0.0, 1.0 - syllable_penalty)

        anchor_component = preserved_anchors / float(anchor_total)
        vowel_component = vowel_similarity_total / float(len(skeletons) or 1)
        syllable_component = syllable_similarity_total / float(len(skeletons) or 1)
        phone_order_component = 1.0
        duration_component = max(0.0, 1.0 - (controls.timing_instability * 0.2))

        score = (
            (anchor_component * weights.anchor_phone_preservation)
            + (vowel_component * weights.vowel_nucleus_similarity)
            + (syllable_component * weights.syllable_count_similarity)
            + (phone_order_component * weights.phone_order_similarity)
            + (duration_component * weights.duration_shape_similarity)
        )
        score = round(score, 4)

        short_word_present = any(skeleton.is_short_word for skeleton in skeletons)
        threshold = (
            self.config.pronunciation_similarity_score.short_word_threshold
            if short_word_present
            else self.config.pronunciation_similarity_score.default_threshold
        )
        if controls.allow_full_gibberish:
            override_mode = self.config.override_modes.get("full_gibberish")
            if override_mode is not None:
                threshold = float(override_mode.values.get("minimum_pronunciation_similarity", threshold))

        return PronunciationSimilarityScore(
            score=score,
            threshold=round(float(threshold), 4),
            passed=score >= float(threshold),
            components=SimilarityComponents(
                anchor_phone_preservation=round(anchor_component, 4),
                vowel_nucleus_similarity=round(vowel_component, 4),
                syllable_count_similarity=round(syllable_component, 4),
                phone_order_similarity=round(phone_order_component, 4),
                duration_shape_similarity=round(duration_component, 4),
            ),
        )

    def _repair_plan(
        self, skeletons: List[PronunciationSkeleton], operations: List[EditOperation], controls: SliderControls
    ) -> tuple[List[EditOperation], List[RepairAction]]:
        repaired = [EditOperation(**operation.__dict__) for operation in operations]
        repairs: List[RepairAction] = []

        for action_name in self.config.pronunciation_similarity_score.repair_order:
            applied = False
            details = ""
            if action_name == "restore_anchor_phones":
                anchor_indices = {index for skeleton in skeletons for index in skeleton.anchor_phone_indices}
                filtered = [
                    operation
                    for operation in repaired
                    if not operation.target_phone_indices or operation.target_phone_indices[0] not in anchor_indices
                ]
                applied = len(filtered) != len(repaired)
                if applied:
                    details = "removed_anchor_touching_operations"
                    repaired = filtered
            elif action_name == "reduce_substitution_distance":
                for operation in repaired:
                    if operation.edit_type == EditType.SUBSTITUTE and operation.replacement_phones:
                        operation.notes = [note for note in operation.notes if not note.startswith("distance=")]
                        operation.notes.append("distance=1")
                        applied = True
                if applied:
                    details = "normalized_substitution_distance"
            elif action_name == "remove_deletions":
                filtered = [
                    operation
                    for operation in repaired
                    if not (operation.edit_type == EditType.SUBSTITUTE and not operation.replacement_phones)
                ]
                applied = len(filtered) != len(repaired)
                if applied:
                    details = "deleted_empty_substitutions"
                    repaired = filtered
            elif action_name == "limit_insertions":
                for operation in repaired:
                    if len(operation.inserted_phones) > 1:
                        operation.inserted_phones = operation.inserted_phones[:1]
                        applied = True
                if applied:
                    details = "trimmed_insertions"
            elif action_name == "reduce_repetitions":
                applied = controls.timing_instability > 0.0
                if applied:
                    details = "timing_repetitions_deferred_to_timing_planner_limits"
            elif action_name == "reduce_timing_fragmentation":
                applied = controls.timing_instability > 0.0
                if applied:
                    details = "duration_jitter_capped_in_timing_planner"
            elif action_name == "rescore":
                details = "rescored_after_repairs"
                applied = True
            repairs.append(RepairAction(name=action_name, applied=applied, details=details))

            score = self._score_plan(skeletons, repaired, controls)
            if score.passed:
                break

        return repaired, repairs

    def _candidate_similarity_threshold(
        self, skeleton: PronunciationSkeleton, controls: SliderControls
    ) -> float:
        threshold = (
            self.config.pronunciation_similarity_score.short_word_threshold
            if skeleton.is_short_word
            else self.config.pronunciation_similarity_score.default_threshold
        )
        if controls.allow_full_gibberish:
            override_mode = self.config.override_modes.get("full_gibberish")
            if override_mode is not None:
                threshold = float(override_mode.values.get("minimum_pronunciation_similarity", threshold))
        guardrails = self._guardrails_for_skeleton(self.config.sliders["jibberish"], skeleton)
        return float(guardrails.get("minimum_pronunciation_similarity", threshold))

    def _candidate_similarity(
        self,
        skeleton: PronunciationSkeleton,
        phones: List[str],
        duration_shape_similarity: float,
    ) -> float:
        return self._evidence_computer.pronunciation_similarity(
            skeleton=skeleton,
            candidate_phones=phones,
            duration_shape_similarity=duration_shape_similarity,
        )

    def _phoneme_distance(
        self, original_phones: List[str], candidate_phones: List[str], skeleton: PronunciationSkeleton
    ) -> float:
        mismatches = abs(len(original_phones) - len(candidate_phones))
        for index in range(min(len(original_phones), len(candidate_phones))):
            if original_phones[index] != candidate_phones[index]:
                mismatches += self._evidence_computer.feature_aware_phone_distance(
                    original_phones[index],
                    candidate_phones[index],
                )
                phone_index = skeleton.phone_indices[index]
                if phone_index in skeleton.anchor_phone_indices:
                    mismatches += 0.75
                if phone_index == skeleton.primary_vowel_index:
                    mismatches += 0.75
        return round(min(1.0, mismatches / float(max(len(original_phones), 1) * 2.25)), 4)

    def _grapheme_distance(self, original: str, candidate: str) -> float:
        return self._evidence_computer.weighted_grapheme_distance(original, candidate)

    def _generate_grapheme_candidates(self, word: str, jibberish_value: float) -> List[str]:
        base = word.lower()
        candidates = {base}
        operations = list(
            self.config.sliders["jibberish"].generated_parameters.get("operations", [])
            if isinstance(self.config.sliders["jibberish"].generated_parameters.get("operations"), list)
            else []
        )
        max_candidates = max(MAX_CANDIDATES_0, int(round(10 + (jibberish_value * 28))))
        vowel_families = {
            "a": ["ah", "aw", "aa"],
            "e": ["eh", "ee", "ae"],
            "i": ["ih", "ee", "iy"],
            "o": ["oh", "oo", "aw", "oa"],
            "u": ["uh", "oo", "yu"],
        }
        if not base:
            return [base]
        if not operations or "letter_repetition" in operations or jibberish_value >= 0.05:
            candidates.add(base + base[-1])
            if len(base) > 1:
                candidates.add(base[:-1] + (base[-1] * 2))
        if not operations or "weak_schwa_like_suffix" in operations or jibberish_value >= 0.15:
            candidates.add(base + "h")
            candidates.add(base + "uh")
        if not operations or "vowel_variant_spelling" in operations or jibberish_value >= 0.25:
            for vowel, variants in vowel_families.items():
                if vowel in base:
                    for variant in variants:
                        candidates.add(base.replace(vowel, variant, 1))
                        if jibberish_value >= 0.55:
                            candidates.add(base.replace(vowel, variant, 2))
        if (not operations or "onset_preserving_mutation" in operations or "soft_consonant_insertion" in operations) and jibberish_value >= 0.35 and len(base) > 1:
            candidates.add(base[0] + "y" + base[1:])
            candidates.add(base[0] + "w" + base[1:])
            candidates.add(base[0] + "l" + base[1:])
            candidates.add(base[0] + "r" + base[1:])
        if (not operations or "pseudo_syllable_expansion" in operations) and jibberish_value >= 0.5:
            candidates.add(base + "-oh")
            candidates.add(base + "-uh")
            candidates.add(base + "-ah")
            if len(base) > 2:
                candidates.add(base + "-ee")
        if (not operations or "rime_preserving_mutation" in operations) and jibberish_value >= 0.65 and len(base) > 2:
            candidates.add(base[0] + base[1] + base[1:] + "oh")
            candidates.add(base[:-1] + base[-1] + "ah")
            candidates.add(base[:-1] + "eh" + base[-1])
        if jibberish_value >= 0.75:
            candidates.add(base + base[-1] + "a")
            candidates.add(base + "ya")
            candidates.add(base + "wa")
            if len(base) > 1:
                candidates.add(base[0] + "a" + base[1:] + "uh")
        if jibberish_value >= 0.85:
            candidates.add(base + "-ayo")
            candidates.add(base + "-owa")
            if len(base) > 2:
                candidates.add(base[:-1] + base[-1] + base[-1] + "oh")
                candidates.add(base[0] + "uh" + base[1:] + "a")
        if jibberish_value >= 0.95 and len(base) > 1:
            candidates.add(base[0] + "a" + base[1:] + "-uh")
            candidates.add(base + "-iya")

        ordered = sorted(candidates, key=lambda item: (self._grapheme_distance(base, item), item))
        return ordered[:max_candidates]

    def _build_trace(
        self,
        skeletons: List[PronunciationSkeleton],
        operations: List[EditOperation],
        selected_candidates: List[CandidateScore],
        distance_evidence: Dict[str, DistanceSignal],
        slider_estimates: Dict[str, SliderEstimate],
        repairs: List[RepairAction],
        similarity: PronunciationSimilarityScore,
        candidates: List[CandidateScore],
        evidence_bundle: DistanceEvidenceBundle,
        traversal_paths: List[object],
    ) -> DecisionTrace:
        primary_skeleton = skeletons[0]
        primary_candidate = selected_candidates[0] if selected_candidates else CandidateScore(
            word=primary_skeleton.word,
            word_index=primary_skeleton.word_index,
            grapheme=primary_skeleton.word,
            phones=list(primary_skeleton.phone_sequence),
            pronunciation_similarity=1.0,
            accepted=True,
        )
        return DecisionTrace(
            target_word=primary_skeleton.word,
            target_graphemes=primary_skeleton.word,
            target_phones=list(primary_skeleton.phone_sequence),
            selected_grapheme_variant=primary_candidate.grapheme,
            selected_phones=list(primary_candidate.phones),
            pronunciation_similarity=similarity.score,
            slider_values={name: estimate.value for name, estimate in slider_estimates.items()},
            confidence={name: estimate.confidence for name, estimate in slider_estimates.items()},
            distance_evidence={name: signal.value for name, signal in distance_evidence.items()},
            distance_features={name: dict(signal.raw_features) for name, signal in distance_evidence.items()},
            routing_rules={
                name: decision.routing_rule or ""
                for name, decision in evidence_bundle.routing.items()
            },
            acoustic_backend=str(
                distance_evidence.get("acoustic_embedding_distance", DistanceSignal(name="acoustic_embedding_distance", value=0.0)).raw_features.get("acoustic_backend", "proxy")
            ),
            acoustic_model_path=str(
                distance_evidence.get("acoustic_embedding_distance", DistanceSignal(name="acoustic_embedding_distance", value=0.0)).raw_features.get("acoustic_model_path", "")
            ),
            acoustic_model_loaded=bool(
                distance_evidence.get("acoustic_embedding_distance", DistanceSignal(name="acoustic_embedding_distance", value=0.0)).raw_features.get("acoustic_model_loaded", False)
            ),
            acoustic_model_compatibility=str(
                distance_evidence.get("acoustic_embedding_distance", DistanceSignal(name="acoustic_embedding_distance", value=0.0)).raw_features.get("acoustic_model_compatibility", "unavailable")
            ),
            traversal_mode=primary_candidate.traversal_source,
            traversal_source=primary_candidate.traversal_source,
            traversal_path=[
                "{0}:{1}:{2}>{3}>{4}".format(
                    path.mode,
                    getattr(path, "proposal_source", ""),
                    state.left,
                    state.center,
                    state.right,
                )
                for path in traversal_paths[:2]
                for state in getattr(path, "states", [])[:6]
            ],
            edits=[
                "{0}:{1}".format(operation.edit_type.value, ",".join(str(index) for index in operation.target_phone_indices))
                for operation in operations
            ],
            rejected_candidates=[candidate.grapheme for candidate in candidates if not candidate.accepted],
            repair_actions=[repair.name for repair in repairs if repair.applied],
        )
