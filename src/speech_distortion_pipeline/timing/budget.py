from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from speech_distortion_pipeline.models import DurationBudget, EditPlan, EditType, PhoneGraph, PhoneNode, TimingPlan


class DurationBudgetManager(Protocol):
    """Allocates local stretching and compensatory shortening under a fixed duration budget."""

    def build(self, graph: PhoneGraph, plan: EditPlan) -> TimingPlan:
        raise NotImplementedError


@dataclass
class HeuristicDurationBudgetManager:
    """Allocates local timing budgets with compensatory shortening.

    Each vowel or consonant lengthening request receives a local window centered
    on the edited phone. Compensation is drawn from neighboring phones in the
    same syllable when possible, then expanded by one phone on each side.
    """

    manager_name: str = "heuristic_duration_budget_manager_v1"
    max_compensation_ratio: float = 0.85

    def build(self, graph: PhoneGraph, plan: EditPlan) -> TimingPlan:
        nodes_by_index = {node.index: node for node in graph.nodes}
        budgets: List[DurationBudget] = []

        for operation in plan.operations:
            if operation.edit_type not in {EditType.LENGTHEN_VOWEL, EditType.LENGTHEN_CONSONANT}:
                continue
            if not operation.target_phone_indices or operation.target_duration_delta_sec <= 0.0:
                continue

            target_index = operation.target_phone_indices[0]
            target_node = nodes_by_index.get(target_index)
            if target_node is None:
                continue

            window_nodes = self._build_window(graph.nodes, target_node)
            if not window_nodes:
                continue

            available = self._available_compensation(window_nodes, target_index)
            requested = float(operation.target_duration_delta_sec)
            compensated = min(requested, round(available * self.max_compensation_ratio, 4))
            budgets.append(
                DurationBudget(
                    window_start_phone_index=window_nodes[0].index,
                    window_end_phone_index=window_nodes[-1].index,
                    requested_delta_sec=requested,
                    compensated_delta_sec=compensated,
                    preserved_total_duration=compensated >= requested,
                )
            )

        return TimingPlan(
            budgets=budgets,
            total_requested_delta_sec=round(sum(item.requested_delta_sec for item in budgets), 4),
            total_compensated_delta_sec=round(sum(item.compensated_delta_sec for item in budgets), 4),
        )

    def _build_window(self, nodes: List[PhoneNode], target_node: PhoneNode) -> List[PhoneNode]:
        same_word = [node for node in nodes if node.word_index == target_node.word_index]
        same_syllable = [
            node for node in same_word if node.syllable_index == target_node.syllable_index
        ]
        if not same_syllable:
            same_syllable = [target_node]

        start_index = min(node.index for node in same_syllable)
        end_index = max(node.index for node in same_syllable)

        expanded_start = max(min(node.index for node in same_word), start_index - 1)
        expanded_end = min(max(node.index for node in same_word), end_index + 1)
        return [node for node in same_word if expanded_start <= node.index <= expanded_end]

    def _available_compensation(self, window_nodes: List[PhoneNode], target_index: int) -> float:
        available = 0.0
        for node in window_nodes:
            if node.index == target_index:
                continue
            duration = self._node_duration(node)
            if self._is_vowel(node):
                available += max(0.0, duration * 0.45)
            elif node.features.manner in {"liquid", "glide", "nasal"}:
                available += max(0.0, duration * 0.35)
            else:
                available += max(0.0, duration * 0.2)
        return round(available, 4)

    def _node_duration(self, node: PhoneNode) -> float:
        if node.start_sec is None or node.end_sec is None:
            return 0.0
        return max(0.0, float(node.end_sec) - float(node.start_sec))

    def _is_vowel(self, node: PhoneNode) -> bool:
        return node.phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER"}
