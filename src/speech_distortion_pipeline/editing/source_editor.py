from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from speech_distortion_pipeline.models import (
    AudioBuffer,
    EditPlan,
    EditType,
    PhoneGraph,
    PhoneNode,
    PronunciationSafePlan,
    TimingInstabilityPlan,
    TimingPlan,
)


class SourceSegmentEditor(Protocol):
    """Applies in-place source-domain distortions and local duration changes."""

    def apply(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: EditPlan, timing: TimingPlan
    ) -> AudioBuffer:
        raise NotImplementedError


class PronunciationSourceEditor(Protocol):
    """Applies pronunciation-safe symbolic edits plus clarity and timing-instability shaping."""

    def apply(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: PronunciationSafePlan, timing: TimingInstabilityPlan
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


@dataclass
class HeuristicPronunciationSourceEditor(HeuristicSourceSegmentEditor):
    """Pronunciation-aware source editor built on top of the bootstrap segment editor."""

    editor_name: str = "heuristic_pronunciation_source_editor_v1"

    def apply(
        self, audio: AudioBuffer, graph: PhoneGraph, plan: PronunciationSafePlan, timing: TimingInstabilityPlan
    ) -> AudioBuffer:
        edited = list(audio.samples)
        node_spans = self._node_spans(graph, len(edited), audio.sample_rate_hz)

        for operation in plan.operations:
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

        edited = self._apply_clarity_plan(edited, plan, graph, node_spans)
        edited = self._apply_timing_instability(edited, graph, timing, node_spans, audio.sample_rate_hz)

        return AudioBuffer(
            samples=self._clamp(edited),
            sample_rate_hz=audio.sample_rate_hz,
            channel_count=audio.channel_count,
            speaker_id=audio.speaker_id,
            metadata={
                **audio.metadata,
                "editor_name": self.editor_name,
                "pronunciation_operation_count": str(len(plan.operations)),
                "clarity_parameter_count": str(len(plan.clarity_plan.parameters)),
                "timing_budget_count": str(len(timing.budgets)),
                "timing_pause_count": str(len(timing.pause_after_phone_indices)),
                "timing_stutter_count": str(len(timing.micro_stutter_phone_indices)),
                "timing_onset_repeat_count": str(len(timing.onset_repeat_phone_indices)),
                "pronunciation_similarity_score": "{0:.4f}".format(plan.similarity.score),
                "pronunciation_similarity_threshold": "{0:.4f}".format(plan.similarity.threshold),
                "trace_selected_grapheme_variant": (
                    plan.trace.selected_grapheme_variant if plan.trace is not None else ""
                ),
                "trace_selected_phones": " ".join(plan.trace.selected_phones) if plan.trace is not None else "",
            },
        )

    def _apply_clarity_plan(
        self,
        samples: List[float],
        plan: PronunciationSafePlan,
        graph: PhoneGraph,
        node_spans: Dict[int, Tuple[int, int]],
    ) -> List[float]:
        output = list(samples)
        params = plan.clarity_plan.parameters
        if not params:
            return output

        blur_ms = float(params.get("transition_blur_ms", 0.0))
        blur_radius = max(1, int(round(blur_ms / 12.0)))
        if blur_radius > 1:
            output = self._moving_average(output, blur_radius)

        high_freq_reduction = float(params.get("high_frequency_reduction", 0.0))
        if high_freq_reduction > 0.0:
            blurred = self._moving_average(output, max(2, blur_radius + 2))
            output = [
                self._clamp_value((sample * (1.0 - high_freq_reduction)) + (smooth * high_freq_reduction))
                for sample, smooth in zip(output, blurred)
            ]

        breath_noise_mix = float(params.get("breath_noise_mix", 0.0))
        if breath_noise_mix > 0.0:
            output = self._inject_breath_noise(output, breath_noise_mix)

        consonant_softening = float(params.get("consonant_burst_softening", 0.0))
        fricative_smearing = float(params.get("fricative_smearing", 0.0))
        for node in graph.nodes:
            span = node_spans.get(node.index)
            if span is None:
                continue
            start, end = span
            segment = output[start:end]
            if not segment:
                continue
            if node.features.manner == "stop" and consonant_softening > 0.0:
                output[start:end] = self._soften_stop(segment, consonant_softening)
            elif node.features.manner == "fricative" and fricative_smearing > 0.0:
                smear_radius = max(2, int(round(3 + (fricative_smearing * 6))))
                output[start:end] = self._moving_average(segment, smear_radius)[: len(segment)]
            elif self._is_vowel(node.phone):
                formant_centralization = float(params.get("formant_centralization", 0.0))
                if formant_centralization > 0.0:
                    output[start:end] = self._centralize_vowel(segment, formant_centralization)
        return output

    def _apply_timing_instability(
        self,
        samples: List[float],
        graph: PhoneGraph,
        timing: TimingInstabilityPlan,
        node_spans: Dict[int, Tuple[int, int]],
        sample_rate_hz: int,
    ) -> List[float]:
        output = list(samples)

        for budget in sorted(timing.budgets, key=lambda item: item.window_start_phone_index, reverse=True):
            window_span = self._window_span(budget.window_start_phone_index, budget.window_end_phone_index, node_spans)
            if window_span is None:
                continue
            start_sample, end_sample = window_span
            window = output[start_sample:end_sample]
            if not window:
                continue
            scale = 1.0 + budget.requested_delta_sec
            target_length = max(1, int(round(len(window) * scale)))
            retimed = self._resample(window, target_length)
            output[start_sample:end_sample] = self._resample(retimed, len(window))

        for phone_index in timing.pause_after_phone_indices:
            span = node_spans.get(phone_index)
            if span is None:
                continue
            pause_len = max(1, int(round(sample_rate_hz * 0.015)))
            start = min(len(output), span[1])
            end = min(len(output), start + pause_len)
            for idx in range(start, end):
                output[idx] *= 0.12

        for phone_index in timing.micro_stutter_phone_indices:
            span = node_spans.get(phone_index)
            if span is None:
                continue
            start, end = span
            segment = output[start:end]
            if not segment:
                continue
            output[start:end] = self._micro_stutter(segment)

        for phone_index in timing.onset_repeat_phone_indices:
            span = node_spans.get(phone_index)
            if span is None:
                continue
            start, end = span
            segment = output[start:end]
            if len(segment) < 6:
                continue
            onset_len = max(2, len(segment) // 5)
            onset = segment[:onset_len]
            repeated = onset + segment[:-onset_len]
            output[start:end] = self._resample(repeated, len(segment))

        return output

    def _inject_breath_noise(self, samples: List[float], mix: float) -> List[float]:
        output: List[float] = []
        state = 0.0
        phase = 0
        for value in samples:
            phase += 1
            pseudo_noise = (((phase * 1103515245) + 12345) % 2048) / 1024.0 - 1.0
            state = (0.93 * state) + (0.07 * pseudo_noise)
            output.append(self._clamp_value((value * (1.0 - mix)) + (state * 0.08 * mix)))
        return output

    def _soften_stop(self, samples: List[float], amount: float) -> List[float]:
        softened = list(samples)
        attack_len = max(1, len(softened) // 5)
        for index in range(attack_len):
            softened[index] *= max(0.25, 1.0 - amount)
        return self._moving_average(softened, 2)

    def _centralize_vowel(self, samples: List[float], amount: float) -> List[float]:
        mean = sum(samples) / float(len(samples) or 1)
        return [self._clamp_value((sample * (1.0 - amount * 0.5)) + (mean * amount * 0.5)) for sample in samples]

    def _micro_stutter(self, samples: List[float]) -> List[float]:
        output: List[float] = []
        gate = max(2, len(samples) // 12)
        for index, value in enumerate(samples):
            if ((index // gate) % 3) == 1:
                output.append(value * 0.2)
            else:
                output.append(value)
        return output

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER", "AA"}
