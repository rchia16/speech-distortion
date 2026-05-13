from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from speech_distortion_pipeline.config import PronunciationSliderConfig
from speech_distortion_pipeline.models import (
    ClarityPlan,
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
    SeverityProfile,
    SimilarityComponents,
    SliderControls,
    WordComplexity,
)


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


class PronunciationPlanner(Protocol):
    """Builds pronunciation-safe symbolic plans from slider controls."""

    def plan(self, graph: PhoneGraph, controls: SliderControls) -> PronunciationSafePlan:
        raise NotImplementedError


@dataclass
class HeuristicPronunciationPlanner:
    config: PronunciationSliderConfig
    planner_name: str = "heuristic_pronunciation_safe_planner_v1"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.global_constraints.random_seed)

    def plan(self, graph: PhoneGraph, controls: SliderControls) -> PronunciationSafePlan:
        skeletons = self._build_skeletons(graph)
        protection_maps = [self._build_protection_map(skeleton) for skeleton in skeletons]
        operations = self._plan_jibberish(graph, skeletons, controls)
        clarity_plan = self._build_clarity_plan(controls)
        generated_parameters = {
            "jibberish": self._derive_jibberish_parameters(controls),
            "clarity": clarity_plan.parameters,
            "timing_instability": self._derive_timing_parameters(controls),
        }

        repairs: List[RepairAction] = []
        similarity = self._score_plan(skeletons, operations, controls)
        if not similarity.passed:
            operations, repairs = self._repair_plan(skeletons, operations, controls)
            similarity = self._score_plan(skeletons, operations, controls)

        operations.sort(key=lambda item: (item.target_phone_indices[0], item.edit_type.value) if item.target_phone_indices else (10**9, item.edit_type.value))
        return PronunciationSafePlan(
            controls=controls,
            skeletons=skeletons,
            protection_maps=protection_maps,
            operations=operations,
            clarity_plan=clarity_plan,
            similarity=similarity,
            repairs=repairs,
            generated_parameters=generated_parameters,
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
        jibberish_guardrails = self.config.sliders["jibberish"].pronunciation_guardrails.values
        return ProtectionMap(
            protected_phone_indices=list(skeleton.anchor_phone_indices),
            anchor_phone_indices=list(skeleton.anchor_phone_indices),
            primary_vowel_index=skeleton.primary_vowel_index,
            preserve_phone_order=bool(jibberish_guardrails.get("preserve_phone_order", True)),
            preserve_syllable_count=bool(jibberish_guardrails.get("preserve_syllable_count", True)),
        )

    def _plan_jibberish(
        self, graph: PhoneGraph, skeletons: List[PronunciationSkeleton], controls: SliderControls
    ) -> List[EditOperation]:
        slider = self.config.sliders["jibberish"]
        parameters = self._derive_jibberish_parameters(controls)
        operations: List[EditOperation] = []
        nodes_by_index = {node.index: node for node in graph.nodes}

        for skeleton in skeletons:
            word_guardrails = self._guardrails_for_skeleton(slider, skeleton)
            candidate_indices = [
                index
                for index in skeleton.phone_indices
                if index not in skeleton.anchor_phone_indices and index != skeleton.primary_vowel_index
            ]
            substitution_budget = int(
                min(
                    len(candidate_indices),
                    round(len(skeleton.phone_indices) * float(word_guardrails.get("max_phone_substitution_ratio", 0.0))),
                )
            )
            substitution_target = min(
                substitution_budget,
                int(round(len(candidate_indices) * parameters["substitution_probability"])),
            )

            for phone_index in candidate_indices[:substitution_target]:
                node = nodes_by_index[phone_index]
                replacement = self._replacement_phone(node, int(parameters["substitution_distance"]))
                operations.append(
                    EditOperation(
                        edit_type=EditType.SUBSTITUTE,
                        target_phone_indices=[phone_index],
                        replacement_phones=[replacement],
                        complexity_weight=round(controls.jibberish, 4),
                        notes=["pronunciation_safe_jibberish", "distance={0}".format(int(parameters["substitution_distance"]))],
                    )
                )

            if controls.allow_full_gibberish and parameters["deletion_probability"] > 0.0 and candidate_indices:
                operations.append(
                    EditOperation(
                        edit_type=EditType.SUBSTITUTE,
                        target_phone_indices=[candidate_indices[-1]],
                        replacement_phones=["AH"],
                        complexity_weight=round(controls.jibberish, 4),
                        notes=["full_gibberish_override"],
                    )
                )

            if not controls.allow_full_gibberish and controls.jibberish > 0.65 and skeleton.is_short_word:
                first_index = skeleton.phone_indices[0]
                operations.append(
                    EditOperation(
                        edit_type=EditType.DISTORT,
                        target_phone_indices=[first_index],
                        complexity_weight=round(controls.jibberish, 4),
                        notes=["short_word_anchor_preserved", "surface_distortion_only"],
                    )
                )

        return operations

    def _guardrails_for_skeleton(self, slider: object, skeleton: PronunciationSkeleton) -> Dict[str, object]:
        assert hasattr(slider, "pronunciation_guardrails")
        base = dict(slider.pronunciation_guardrails.values)
        short_word_mode = slider.pronunciation_guardrails.short_word_mode
        if skeleton.is_short_word and short_word_mode is not None:
            base["max_phone_substitution_ratio"] = short_word_mode.max_phone_substitution_ratio
            base["max_deletions_per_word"] = short_word_mode.max_deletions_per_word
            base["preserve_first_phone"] = short_word_mode.preserve_first_phone
            base["preserve_primary_vowel_nucleus"] = short_word_mode.preserve_primary_vowel_nucleus
        return base

    def _replacement_phone(self, node: PhoneNode, distance: int) -> str:
        manner = node.features.manner
        if self._is_vowel(node.phone):
            ladder = ["AH", "EH", "IH", "OW", "ER", "AE"]
        elif manner == "nasal":
            ladder = ["N", "M", "NG"]
        elif manner == "stop":
            ladder = ["T", "D", "K", "P"]
        elif manner == "fricative":
            ladder = ["S", "SH", "Z", "TH"]
        elif manner == "liquid":
            ladder = ["R", "L", "W"]
        else:
            ladder = [node.phone]
        if node.phone in ladder:
            start = ladder.index(node.phone)
        else:
            start = 0
        return ladder[min(start + max(distance - 1, 0), len(ladder) - 1)]

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER", "AA"}

    def _derive_jibberish_parameters(self, controls: SliderControls) -> Dict[str, float]:
        guardrails = self.config.sliders["jibberish"].pronunciation_guardrails.values
        return {
            "substitution_probability": round(pow(controls.jibberish, 1.2) * 0.60, 4),
            "substitution_distance": float(round(1 + (controls.jibberish * 3))),
            "vowel_shift_probability": round(max(min((controls.jibberish - 0.15) / 0.85, 1.0), 0.0) * 0.50, 4),
            "insertion_probability": round(max(min((controls.jibberish - 0.45) / 0.55, 1.0), 0.0) * 0.30, 4),
            "deletion_probability": 0.0 if not controls.allow_full_gibberish else round(controls.jibberish * 0.15, 4),
            "repetition_probability": round(max(min((controls.jibberish - 0.60) / 0.40, 1.0), 0.0) * 0.25, 4),
            "max_anchor_phone_distance": float(guardrails.get("max_anchor_phone_distance", 1)),
        }

    def _derive_timing_parameters(self, controls: SliderControls) -> Dict[str, float]:
        guardrails = self.config.sliders["timing_instability"].pronunciation_guardrails.values
        max_pause = float(guardrails.get("max_pause_ms_inside_short_word", 180))
        return {
            "phone_duration_jitter": round(controls.timing_instability * 0.35, 4),
            "syllable_duration_jitter": round(controls.timing_instability * 0.25, 4),
            "pause_insertion_probability": round(max(min((controls.timing_instability - 0.30) / 0.70, 1.0), 0.0) * 0.35, 4),
            "pause_duration_ms": round(min(20 + (controls.timing_instability * 180), max_pause), 4),
            "micro_stutter_probability": round(max(min((controls.timing_instability - 0.45) / 0.55, 1.0), 0.0) * 0.45, 4),
            "onset_repeat_probability": round(max(min((controls.timing_instability - 0.55) / 0.45, 1.0), 0.0) * 0.35, 4),
        }

    def _build_clarity_plan(self, controls: SliderControls) -> ClarityPlan:
        guardrails = dict(self.config.sliders["clarity"].pronunciation_guardrails.values)
        max_formant = float(guardrails.get("max_formant_centralization_for_anchor_vowels", 0.45))
        min_burst = float(guardrails.get("min_consonant_burst_energy_ratio", 0.35))
        max_blur_ms = float(guardrails.get("max_transition_blur_ms", 60))
        return ClarityPlan(
            parameters={
                "formant_centralization": round(min(controls.clarity * 0.70, max_formant), 4),
                "spectral_tilt_db_per_octave": round(controls.clarity * -5.0, 4),
                "high_frequency_reduction": round(controls.clarity * 0.80, 4),
                "consonant_burst_softening": round(min(pow(controls.clarity, 1.1) * 0.90, 1 - min_burst), 4),
                "fricative_smearing": round(controls.clarity * 0.85, 4),
                "transition_blur_ms": round(min(5 + (controls.clarity * 55), max_blur_ms), 4),
                "breath_noise_mix": round(controls.clarity * 0.12, 4),
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
            self.config.global_constraints.minimum_pronunciation_similarity_short_word
            if short_word_present
            else self.config.global_constraints.minimum_pronunciation_similarity
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
