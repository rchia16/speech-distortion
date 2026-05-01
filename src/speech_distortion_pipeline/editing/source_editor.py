from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from speech_distortion_pipeline.models import AudioBuffer, EditPlan, EditType, PhoneGraph, PhoneNode, TimingPlan


class SourceSegmentEditor(Protocol):
    """Applies in-place source-domain distortions and local duration changes."""

    def apply(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> AudioBuffer:
        raise NotImplementedError


@dataclass
class HeuristicSourceSegmentEditor:
    """Dependency-free source-domain editor for early bootstrap slices."""

    editor_name: str = "heuristic_source_segment_editor_v1"

    def apply(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> AudioBuffer:
        edited = list(audio.samples)
        node_spans = self._node_spans(graph, len(edited), audio.sample_rate_hz)

        for operation in plan.operations:
            if operation.edit_type in {EditType.LENGTHEN_VOWEL, EditType.LENGTHEN_CONSONANT}:
                continue
            if not operation.target_phone_indices:
                continue
            target_index = operation.target_phone_indices[0]
            span = node_spans.get(target_index)
            if span is None:
                continue
            segment = edited[span[0] : span[1]]
            if not segment:
                continue
            transformed = self._transform_segment(segment, operation.edit_type)
            edited[span[0] : span[1]] = transformed[: len(segment)]

        grouped = self._group_timing_budgets(graph, plan, timing)
        for key in sorted(grouped.keys(), reverse=True):
            window_start, window_end = key
            payload = grouped[key]
            window_span = self._window_span(window_start, window_end, node_spans)
            if window_span is None:
                continue
            start_sample, end_sample = window_span
            window_samples = edited[start_sample:end_sample]
            nodes = payload["nodes"]
            deltas = payload["deltas"]
            reconstructed = self._retime_window(window_samples, nodes, deltas, node_spans, start_sample)
            edited[start_sample:end_sample] = reconstructed

        return AudioBuffer(
            samples=self._clamp(edited),
            sample_rate_hz=audio.sample_rate_hz,
            channel_count=audio.channel_count,
            speaker_id=audio.speaker_id,
            metadata={
                **audio.metadata,
                "editor_name": self.editor_name,
                "edit_operation_count": str(len(plan.operations)),
                "timing_budget_count": str(len(timing.budgets)),
            },
        )

    def _node_spans(
        self, graph: PhoneGraph, total_samples: int, sample_rate_hz: int
    ) -> Dict[int, Tuple[int, int]]:
        spans: Dict[int, Tuple[int, int]] = {}
        for node in graph.nodes:
            if node.start_sec is None or node.end_sec is None:
                continue
            start = max(0, min(total_samples, int(round(node.start_sec * sample_rate_hz))))
            end = max(start + 1, min(total_samples, int(round(node.end_sec * sample_rate_hz))))
            spans[node.index] = (start, end)
        return spans

    def _transform_segment(self, samples: List[float], edit_type: EditType) -> List[float]:
        if edit_type == EditType.DISTORT:
            return self._soft_clip(self._moving_average(samples, 5), 0.65)
        if edit_type == EditType.SUBSTITUTE:
            blurred = self._moving_average(samples, 7)
            return [self._clamp_value(value * 0.55) for value in blurred]
        if edit_type == EditType.DISTORTED_SUBSTITUTE:
            base = self._soft_clip(self._moving_average(samples, 9), 0.5)
            return [
                self._clamp_value(value * (0.7 if (index // 8) % 2 == 0 else -0.25))
                for index, value in enumerate(base)
            ]
        if edit_type == EditType.ADD:
            shifted = [0.0] * len(samples)
            offset = max(1, min(len(samples) // 5, 160))
            for index, value in enumerate(samples):
                mixed = 0.65 * value
                if index >= offset:
                    mixed += 0.35 * samples[index - offset]
                shifted[index] = self._clamp_value(mixed)
            return shifted
        return samples

    def _group_timing_budgets(
        self, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> Dict[Tuple[int, int], Dict[str, object]]:
        nodes_by_index = {node.index: node for node in graph.nodes}
        grouped: Dict[Tuple[int, int], Dict[str, object]] = {}
        for budget in timing.budgets:
            key = (budget.window_start_phone_index, budget.window_end_phone_index)
            payload = grouped.setdefault(key, {"nodes": [], "deltas": {}})
            nodes = payload["nodes"]
            deltas = payload["deltas"]
            if not nodes:
                payload["nodes"] = [
                    nodes_by_index[index]
                    for index in sorted(nodes_by_index)
                    if budget.window_start_phone_index <= index <= budget.window_end_phone_index
                ]
                nodes = payload["nodes"]
            for operation in plan.operations:
                if operation.edit_type not in {EditType.LENGTHEN_VOWEL, EditType.LENGTHEN_CONSONANT}:
                    continue
                if not operation.target_phone_indices:
                    continue
                target = operation.target_phone_indices[0]
                if not (budget.window_start_phone_index <= target <= budget.window_end_phone_index):
                    continue
                deltas[target] = deltas.get(target, 0.0) + float(operation.target_duration_delta_sec)
        return grouped

    def _window_span(
        self, start_index: int, end_index: int, node_spans: Dict[int, Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if start_index not in node_spans or end_index not in node_spans:
            return None
        return (node_spans[start_index][0], node_spans[end_index][1])

    def _retime_window(
        self,
        window_samples: List[float],
        nodes: List[PhoneNode],
        deltas: Dict[int, float],
        node_spans: Dict[int, Tuple[int, int]],
        window_start_sample: int,
    ) -> List[float]:
        if not nodes:
            return window_samples

        original_lengths: Dict[int, int] = {}
        desired_lengths: Dict[int, int] = {}
        available_pool = 0
        for node in nodes:
            span = node_spans.get(node.index)
            if span is None:
                continue
            local_start = max(0, span[0] - window_start_sample)
            local_end = max(local_start + 1, span[1] - window_start_sample)
            length = max(1, local_end - local_start)
            original_lengths[node.index] = length
            requested = int(round(deltas.get(node.index, 0.0) * (span[1] - span[0]) / max(node.end_sec - node.start_sec, 1e-6))) if (node.start_sec is not None and node.end_sec is not None and deltas.get(node.index, 0.0) > 0.0) else 0
            if requested == 0 and deltas.get(node.index, 0.0) > 0.0:
                requested = max(1, int(round(deltas[node.index] * 24000)))
            desired_lengths[node.index] = length + requested
            if node.index not in deltas:
                available_pool += max(0, length - self._minimum_length(node))

        extra_needed = sum(max(0, desired_lengths[index] - original_lengths[index]) for index in desired_lengths)
        if extra_needed > 0 and available_pool > 0:
            for node in nodes:
                if node.index in deltas:
                    continue
                shrink_cap = max(0, original_lengths[node.index] - self._minimum_length(node))
                take = int(round(extra_needed * (float(shrink_cap) / float(available_pool)))) if available_pool else 0
                desired_lengths[node.index] = max(self._minimum_length(node), original_lengths[node.index] - take)

        total_original = sum(original_lengths.values())
        total_desired = sum(desired_lengths.values())
        difference = total_desired - total_original
        if difference != 0:
            adjustable = [node for node in reversed(nodes) if desired_lengths[node.index] > self._minimum_length(node)]
            for node in adjustable:
                if difference == 0:
                    break
                step = min(abs(difference), max(0, desired_lengths[node.index] - self._minimum_length(node))) if difference > 0 else abs(difference)
                if difference > 0:
                    desired_lengths[node.index] -= step
                    difference -= step
                else:
                    desired_lengths[node.index] += step
                    difference += step

        rebuilt: List[float] = []
        for node in nodes:
            span = node_spans.get(node.index)
            if span is None:
                continue
            local_start = max(0, span[0] - window_start_sample)
            local_end = max(local_start + 1, span[1] - window_start_sample)
            segment = window_samples[local_start:local_end]
            rebuilt.extend(self._resample(segment, desired_lengths[node.index]))

        if len(rebuilt) != len(window_samples):
            rebuilt = self._resample(rebuilt, len(window_samples))
        return rebuilt

    def _minimum_length(self, node: PhoneNode) -> int:
        if node.phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER"}:
            return 8
        return 4

    def _moving_average(self, samples: List[float], radius: int) -> List[float]:
        if radius <= 1 or len(samples) < 3:
            return list(samples)
        output: List[float] = []
        for index in range(len(samples)):
            start = max(0, index - radius)
            end = min(len(samples), index + radius + 1)
            window = samples[start:end]
            output.append(sum(window) / float(len(window)))
        return output

    def _soft_clip(self, samples: List[float], threshold: float) -> List[float]:
        output: List[float] = []
        for value in samples:
            if value > threshold:
                value = threshold + ((value - threshold) * 0.25)
            elif value < -threshold:
                value = -threshold + ((value + threshold) * 0.25)
            output.append(self._clamp_value(value))
        return output

    def _resample(self, samples: List[float], target_length: int) -> List[float]:
        if target_length <= 0:
            return []
        if not samples:
            return [0.0] * target_length
        if len(samples) == target_length:
            return list(samples)
        if len(samples) == 1:
            return [samples[0]] * target_length
        output: List[float] = []
        scale = float(len(samples) - 1) / float(max(target_length - 1, 1))
        for index in range(target_length):
            position = index * scale
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            mix = position - left
            output.append((samples[left] * (1.0 - mix)) + (samples[right] * mix))
        return output

    def _clamp(self, samples: List[float]) -> List[float]:
        return [self._clamp_value(value) for value in samples]

    def _clamp_value(self, value: float) -> float:
        return max(-1.0, min(1.0, value))
