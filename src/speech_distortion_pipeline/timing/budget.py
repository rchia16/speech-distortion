from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from speech_distortion_pipeline.config import HybridConfig
from speech_distortion_pipeline.models import (
    DurationBudget,
    EditPlan,
    EditType,
    PhoneGraph,
    PhoneNode,
    PronunciationSafePlan,
    TimingInstabilityPlan,
    TimingPlan,
)


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


class PronunciationTimingPlanner(Protocol):
    """Builds timing-instability directives under pronunciation-preserving guardrails."""

    def build(self, graph: PhoneGraph, plan: PronunciationSafePlan) -> TimingInstabilityPlan:
        raise NotImplementedError


@dataclass
class HeuristicPronunciationTimingPlanner:
    config: HybridConfig
    planner_name: str = "heuristic_pronunciation_timing_planner_v1"

    def build(self, graph: PhoneGraph, plan: PronunciationSafePlan) -> TimingInstabilityPlan:
        estimate = plan.slider_estimates.get("timing_instability")
        routed_value = estimate.value if estimate is not None else plan.controls.timing_instability
        timing_params = plan.generated_parameters.get("timing_instability", {})
        guardrails = self.config.sliders["timing_instability"].guardrails.values
        by_word: Dict[int, List[PhoneNode]] = {}
        for node in graph.nodes:
            by_word.setdefault(node.word_index, []).append(node)

        budgets: List[DurationBudget] = []
        pause_after: List[int] = []
        onset_repeats: List[int] = []
        micro_stutters: List[int] = []

        total_requested = 0.0
        total_compensated = 0.0
        max_drift = float(
            guardrails.get(
                "max_total_duration_drift_ratio",
                self.config.global_constraints.max_total_duration_drift_ratio,
            )
        )
        duration_scale = max_drift * routed_value

        for skeleton in plan.skeletons:
            nodes = by_word.get(skeleton.word_index, [])
            if not nodes:
                continue
            start_index = nodes[0].index
            end_index = nodes[-1].index
            word_duration = sum(self._node_duration(node) for node in nodes)
            duration_jitter = float(timing_params.get("phone_duration_jitter", routed_value * 0.35))
            syllable_jitter = float(timing_params.get("syllable_duration_jitter", routed_value * 0.25))
            requested = round(word_duration * min(max_drift, duration_scale + (duration_jitter * 0.08) + (syllable_jitter * 0.06)), 4)
            compensated = round(min(requested, word_duration * 0.8), 4)
            budgets.append(
                DurationBudget(
                    window_start_phone_index=start_index,
                    window_end_phone_index=end_index,
                    requested_delta_sec=requested,
                    compensated_delta_sec=compensated,
                    preserved_total_duration=compensated >= requested,
                )
            )
            total_requested += requested
            total_compensated += compensated

            micro_probability = float(timing_params.get("micro_stutter_probability", max(0.0, routed_value - 0.45)))
            onset_probability = float(timing_params.get("onset_repeat_probability", max(0.0, routed_value - 0.55)))
            pause_probability = float(timing_params.get("pause_insertion_probability", max(0.0, routed_value - 0.30)))

            if micro_probability > 0.0:
                micro_stutters.append(start_index)
            if onset_probability > 0.0:
                repeat_limit = int(guardrails.get("max_onset_repeats", 2))
                repeat_count = min(repeat_limit, max(1, int(round(onset_probability * repeat_limit))))
                onset_repeats.extend([start_index] * repeat_count)
            if pause_probability > 0.0:
                pause_after.append(end_index)

        return TimingInstabilityPlan(
            budgets=budgets,
            pause_after_phone_indices=pause_after,
            onset_repeat_phone_indices=onset_repeats,
            micro_stutter_phone_indices=micro_stutters,
            total_requested_delta_sec=round(total_requested, 4),
            total_compensated_delta_sec=round(total_compensated, 4),
            max_total_duration_drift_ratio=max_drift,
            preserve_phone_order=bool(guardrails.get("preserve_phone_order", True)),
        )

    def _node_duration(self, node: PhoneNode) -> float:
        if node.start_sec is None or node.end_sec is None:
            return 0.0
        return max(0.0, float(node.end_sec) - float(node.start_sec))
