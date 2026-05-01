from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from speech_distortion_pipeline.models import EditOperation, EditPlan, EditType, PhoneGraph, PhoneNode, SeverityProfile, WordComplexity


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
