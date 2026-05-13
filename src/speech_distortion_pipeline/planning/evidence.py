from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from speech_distortion_pipeline.config import HybridConfig
from speech_distortion_pipeline.models import (
    AudioBuffer,
    CandidateScore,
    DistanceEvidenceBundle,
    DistanceSignal,
    PhoneFeatureBundle,
    PhoneGraph,
    PhoneNode,
    PronunciationSkeleton,
    RoutingDecision,
    SliderControls,
    SliderEstimate,
)
from speech_distortion_pipeline.phonology.phone_distance_reference import (
    ranked_candidate_entries_from_reference,
)


@dataclass
class AcousticBackendStatus:
    backend_name: str
    model_path: str
    compatibility: str
    loaded: bool
    load_error: Optional[str] = None
    model: object | None = None
    torch_module: object | None = None
    torchaudio_module: object | None = None


@dataclass
class HeuristicDistanceEvidenceComputer:
    config: HybridConfig

    def __post_init__(self) -> None:
        self._acoustic_backend_status: Optional[AcousticBackendStatus] = None
        self._default_hubert_path = (
            Path(__file__).resolve().parents[1] / "models" / "hubert_base_ls960.pt"
        )

    def compute(
        self,
        graph: PhoneGraph,
        skeletons: List[PronunciationSkeleton],
        candidates_by_word: Dict[int, List[CandidateScore]],
        controls: SliderControls,
        audio: Optional[AudioBuffer] = None,
    ) -> DistanceEvidenceBundle:
        word_audio_features = self._extract_word_audio_features(audio, graph, skeletons)
        signals = self._build_signals(graph, skeletons, candidates_by_word, controls, word_audio_features)
        routing = self._route_signals(signals, controls)
        return DistanceEvidenceBundle(signals=signals, routing=routing)

    def estimate_sliders(self, bundle: DistanceEvidenceBundle) -> Dict[str, SliderEstimate]:
        return {
            name: SliderEstimate(
                value=decision.value,
                confidence=decision.confidence,
                evidence_sources=list(decision.evidence_sources),
                routing_rule=decision.routing_rule,
            )
            for name, decision in bundle.routing.items()
        }

    def feature_aware_phone_distance(self, original_phone: str, candidate_phone: str) -> float:
        if original_phone == candidate_phone:
            return 0.0

        entries = ranked_candidate_entries_from_reference(original_phone)
        if entries:
            best = None
            max_distance = max(float(entry["distance"]) for entry in entries) or 1.0
            for entry in entries:
                phones = [part for part in entry["phones"] if isinstance(part, str)]
                if len(phones) == 1 and phones[0] == candidate_phone:
                    best = float(entry["distance"]) / float(max_distance)
                    break
            if best is not None:
                return round(max(0.0, min(1.0, best)), 4)

        left = self._feature_bundle(original_phone)
        right = self._feature_bundle(candidate_phone)
        penalties = 0.0
        weights = 0.0
        for attr, weight in (
            ("place", 0.18),
            ("manner", 0.28),
            ("voicing", 0.12),
            ("vowel_height", 0.20),
            ("vowel_backness", 0.14),
            ("rounding", 0.08),
        ):
            left_value = getattr(left, attr)
            right_value = getattr(right, attr)
            if left_value is None and right_value is None:
                continue
            weights += weight
            penalties += 0.0 if left_value == right_value else weight
        if weights <= 0.0:
            return 1.0
        return round(max(0.0, min(1.0, penalties / weights)), 4)

    def weighted_grapheme_distance(self, original: str, candidate: str) -> float:
        left = re.sub(r"[^a-z]", "", original.lower())
        right = re.sub(r"[^a-z]", "", candidate.lower())
        if left == right:
            return 0.0
        rows = len(left) + 1
        cols = len(right) + 1
        matrix = [[0.0] * cols for _ in range(rows)]
        for row in range(rows):
            matrix[row][0] = float(row)
        for col in range(cols):
            matrix[0][col] = float(col)
        vowels = {"a", "e", "i", "o", "u", "y"}
        for row in range(1, rows):
            for col in range(1, cols):
                left_char = left[row - 1]
                right_char = right[col - 1]
                if left_char == right_char:
                    substitution_cost = 0.0
                elif left_char in vowels and right_char in vowels:
                    substitution_cost = 0.65
                else:
                    substitution_cost = 1.0
                matrix[row][col] = min(
                    matrix[row - 1][col] + 1.0,
                    matrix[row][col - 1] + 1.0,
                    matrix[row - 1][col - 1] + substitution_cost,
                )

        score = matrix[-1][-1] / float(max(len(left), len(right), 1))
        if left and right and left[0] != right[0]:
            score += 0.25
        if left and right and self._rime(left) != self._rime(right):
            score += 0.15
        if re.search(r"(.)\1", right):
            score += 0.08
        return round(max(0.0, min(1.0, score)), 4)

    def pronunciation_similarity(
        self,
        skeleton: PronunciationSkeleton,
        candidate_phones: List[str],
        duration_shape_similarity: float,
    ) -> float:
        weights = self.config.pronunciation_similarity_score.weights
        anchor_matches = 0
        for anchor_index in skeleton.anchor_phone_indices:
            relative_index = skeleton.phone_indices.index(anchor_index)
            if relative_index < len(candidate_phones) and candidate_phones[relative_index] == skeleton.phone_sequence[relative_index]:
                anchor_matches += 1
        anchor_component = anchor_matches / float(len(skeleton.anchor_phone_indices) or 1)

        vowel_component = 1.0
        if skeleton.primary_vowel_index is not None:
            vowel_component = 0.0
            vowel_position = skeleton.phone_indices.index(skeleton.primary_vowel_index)
            if vowel_position < len(candidate_phones):
                candidate_phone = candidate_phones[vowel_position]
                if candidate_phone == skeleton.primary_vowel_phone:
                    vowel_component = 1.0
                elif self._is_vowel(candidate_phone):
                    vowel_component = 1.0 - min(
                        1.0,
                        self.feature_aware_phone_distance(skeleton.primary_vowel_phone or "", candidate_phone),
                    )

        syllable_component = max(
            0.0,
            1.0 - (abs(len(candidate_phones) - len(skeleton.phone_sequence)) / float(max(len(skeleton.phone_sequence), 1))),
        )
        phone_order_component = self._phone_order_similarity(skeleton.phone_sequence, candidate_phones)
        score = (
            (anchor_component * weights.anchor_phone_preservation)
            + (vowel_component * weights.vowel_nucleus_similarity)
            + (syllable_component * weights.syllable_count_similarity)
            + (phone_order_component * weights.phone_order_similarity)
            + (duration_shape_similarity * weights.duration_shape_similarity)
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _build_signals(
        self,
        graph: PhoneGraph,
        skeletons: List[PronunciationSkeleton],
        candidates_by_word: Dict[int, List[CandidateScore]],
        controls: SliderControls,
        word_audio_features: Dict[int, Dict[str, Any]],
    ) -> Dict[str, DistanceSignal]:
        candidate_features = self._candidate_features(skeletons, candidates_by_word)
        audio_features = self._aggregate_audio_features(word_audio_features)

        signals = {
            "grapheme_distance": DistanceSignal(
                name="grapheme_distance",
                value=round(candidate_features["grapheme_distance"], 4),
                sources=["candidate_generation", "weighted_levenshtein"],
                raw_features={
                    "mutated_candidate_rate": candidate_features["mutated_candidate_rate"],
                    "top_mutated_grapheme_distance": candidate_features["grapheme_distance"],
                },
                aligned_region="word_candidates",
            ),
            "phoneme_distance": DistanceSignal(
                name="phoneme_distance",
                value=round(candidate_features["phoneme_distance"], 4),
                sources=["candidate_g2p", "feature_aware_phone_distance"],
                raw_features={
                    "accepted_candidate_count": candidate_features["accepted_candidate_count"],
                    "best_candidate_phoneme_distance": candidate_features["phoneme_distance"],
                },
                aligned_region="word_candidates",
            ),
            "pronunciation_similarity": DistanceSignal(
                name="pronunciation_similarity",
                value=round(candidate_features["pronunciation_similarity"], 4),
                sources=["candidate_g2p", "skeleton_similarity"],
                raw_features={
                    "best_candidate_similarity": candidate_features["pronunciation_similarity"],
                    "duration_shape_similarity": candidate_features["duration_shape_similarity"],
                },
                aligned_region="word_candidates",
            ),
            "acoustic_embedding_distance": DistanceSignal(
                name="acoustic_embedding_distance",
                value=round(audio_features["acoustic_embedding_distance"], 4),
                sources=self._acoustic_signal_sources(audio_features),
                raw_features=dict(audio_features),
                aligned_region="utterance_audio",
            ),
            "spectral_feature_distance": DistanceSignal(
                name="spectral_feature_distance",
                value=round(audio_features["spectral_feature_distance"], 4),
                sources=["audio_window", "hf_contrast_proxy"],
                raw_features=dict(audio_features),
                aligned_region="utterance_audio",
            ),
            "formant_trajectory_distance": DistanceSignal(
                name="formant_trajectory_distance",
                value=round(audio_features["formant_trajectory_distance"], 4),
                sources=["audio_window", "vowel_centroid_proxy"],
                raw_features=dict(audio_features),
                aligned_region="utterance_audio",
            ),
            "temporal_distance": DistanceSignal(
                name="temporal_distance",
                value=round(audio_features["temporal_distance"], 4),
                sources=["timing_window", "pause_ratio", "repeat_similarity"],
                raw_features=dict(audio_features),
                aligned_region="utterance_timing",
            ),
            "dtw_warp_distance": DistanceSignal(
                name="dtw_warp_distance",
                value=round(audio_features["dtw_warp_distance"], 4),
                sources=["timing_window", "frame_envelope_jitter"],
                raw_features=dict(audio_features),
                aligned_region="utterance_timing",
            ),
            "duration_shape_distance": DistanceSignal(
                name="duration_shape_distance",
                value=round(audio_features["duration_shape_distance"], 4),
                sources=["phone_durations", "duration_ratio"],
                raw_features=dict(audio_features),
                aligned_region="utterance_timing",
            ),
        }

        neural_override = controls.distance_overrides.get("neural_embedding_distance")
        signals["neural_embedding_distance"] = DistanceSignal(
            name="neural_embedding_distance",
            value=round(max(0.0, min(1.0, float(neural_override or 0.0))), 4),
            sources=["override"] if neural_override is not None else ["unavailable"],
            raw_features={"provided": 1.0 if neural_override is not None else 0.0},
            aligned_region="template_space" if neural_override is not None else None,
        )
        return signals

    def _route_signals(
        self,
        signals: Dict[str, DistanceSignal],
        controls: SliderControls,
    ) -> Dict[str, RoutingDecision]:
        routing_scores = {
            "jibberish": {"score": controls.jibberish, "rule": "control_prior", "evidence": []},
            "clarity": {"score": controls.clarity, "rule": "control_prior", "evidence": []},
            "timing_instability": {"score": controls.timing_instability, "rule": "control_prior", "evidence": []},
        }
        for rule in self.config.mismatch_decomposition.routing_rules:
            score, evidence_sources = self._routing_rule_score(rule.evidence, signals)
            if rule.output_slider is None:
                continue
            prior = routing_scores[rule.output_slider]["score"]
            combined = max(prior * 0.35, score * 0.65)
            if combined >= routing_scores[rule.output_slider]["score"]:
                routing_scores[rule.output_slider] = {
                    "score": round(max(0.0, min(1.0, combined)), 4),
                    "rule": rule.name,
                    "evidence": evidence_sources,
                }

        decisions: Dict[str, RoutingDecision] = {}
        for slider_name, payload in routing_scores.items():
            evidence_values = [
                signals[source].value if source in signals else 0.0
                for source in payload["evidence"]
            ]
            confidence = self._confidence_for_evidence(evidence_values)
            value = float(payload["score"])
            if confidence < 0.65:
                value *= 0.8
            if slider_name == "jibberish":
                dtw_only = signals["dtw_warp_distance"].value >= 0.7 and signals["phoneme_distance"].value <= 0.25
                neural_only = signals["neural_embedding_distance"].value >= 0.7 and signals["phoneme_distance"].value <= 0.25
                if dtw_only or neural_only:
                    value = min(value, 0.35)
                    confidence = min(confidence, 0.62)
            decisions[slider_name] = RoutingDecision(
                slider_name=slider_name,
                value=round(max(0.0, min(1.0, value)), 4),
                confidence=confidence,
                evidence_sources=list(payload["evidence"]),
                routing_rule=str(payload["rule"]),
            )
        return decisions

    def _routing_rule_score(
        self,
        evidence_names: Iterable[str],
        signals: Dict[str, DistanceSignal],
    ) -> tuple[float, List[str]]:
        scores: List[float] = []
        sources: List[str] = []
        for name in evidence_names:
            score, source_name = self._rule_evidence_score(name, signals)
            if source_name is None:
                continue
            scores.append(score)
            sources.append(source_name)
        if not scores:
            return 0.0, []
        return round(sum(scores) / float(len(scores)), 4), sources

    def _rule_evidence_score(
        self,
        evidence_name: str,
        signals: Dict[str, DistanceSignal],
    ) -> tuple[float, Optional[str]]:
        mapping = {
            "high_phoneme_distance": ("phoneme_distance", False),
            "low_pronunciation_similarity": ("pronunciation_similarity", True),
            "sustained_phone_level_residual": ("phoneme_distance", False),
            "high_acoustic_embedding_distance": ("acoustic_embedding_distance", False),
            "high_spectral_feature_distance": ("spectral_feature_distance", False),
            "stable_phone_identity": ("phoneme_distance", True),
            "high_temporal_distance": ("temporal_distance", False),
            "abnormal_dtw_warp_ratio": ("dtw_warp_distance", False),
            "pause_or_hold_mismatch": ("temporal_distance", False),
            "high_neural_embedding_distance": ("neural_embedding_distance", False),
            "high_wasserstein_distribution_distance": ("neural_embedding_distance", False),
        }
        mapped = mapping.get(evidence_name)
        if mapped is None:
            return 0.0, None
        signal_name, invert = mapped
        signal = signals.get(signal_name)
        if signal is None:
            return 0.0, None
        value = 1.0 - signal.value if invert else signal.value
        return round(max(0.0, min(1.0, value)), 4), signal_name

    def _candidate_features(
        self,
        skeletons: List[PronunciationSkeleton],
        candidates_by_word: Dict[int, List[CandidateScore]],
    ) -> Dict[str, float]:
        mutated_distances: List[float] = []
        phoneme_distances: List[float] = []
        similarities: List[float] = []
        accepted_count = 0.0
        mutated_count = 0.0
        total_count = 0.0
        for skeleton in skeletons:
            candidates = candidates_by_word.get(skeleton.word_index, [])
            if not candidates:
                continue
            total_count += float(len(candidates))
            mutated = [candidate for candidate in candidates if candidate.grapheme != skeleton.word]
            if mutated:
                mutated_count += float(len(mutated))
                mutated_distances.append(min(candidate.grapheme_distance for candidate in mutated))
            accepted = [candidate for candidate in candidates if candidate.accepted]
            if accepted:
                accepted_count += float(len(accepted))
                phoneme_distances.append(min(candidate.phoneme_distance for candidate in accepted))
                similarities.append(max(candidate.pronunciation_similarity for candidate in accepted))
            else:
                top = candidates[0]
                phoneme_distances.append(top.phoneme_distance)
                similarities.append(top.pronunciation_similarity)
        return {
            "grapheme_distance": self._mean(mutated_distances, 0.0),
            "phoneme_distance": self._mean(phoneme_distances, 0.0),
            "pronunciation_similarity": self._mean(similarities, 1.0),
            "mutated_candidate_rate": (mutated_count / total_count) if total_count > 0 else 0.0,
            "accepted_candidate_count": accepted_count,
            "duration_shape_similarity": max(0.0, 1.0 - self._mean(phoneme_distances, 0.0)),
        }

    def _aggregate_audio_features(self, per_word: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        if not per_word:
            metadata = self._acoustic_backend_metadata()
            return {
                "acoustic_embedding_distance": 0.0,
                "spectral_feature_distance": 0.0,
                "formant_trajectory_distance": 0.0,
                "temporal_distance": 0.0,
                "dtw_warp_distance": 0.0,
                "duration_shape_distance": 0.0,
                "pause_ratio": 0.0,
                "repeat_similarity": 0.0,
                "envelope_jitter": 0.0,
                "duration_ratio_deviation": 0.0,
                **metadata,
            }

        values = list(per_word.values())
        acoustic = self._mean([item["acoustic_embedding_distance"] for item in values], 0.0)
        spectral = self._mean([item["spectral_feature_distance"] for item in values], 0.0)
        formant = self._mean([item["formant_trajectory_distance"] for item in values], 0.0)
        temporal = self._mean([item["temporal_distance"] for item in values], 0.0)
        dtw = self._mean([item["dtw_warp_distance"] for item in values], 0.0)
        duration = self._mean([item["duration_shape_distance"] for item in values], 0.0)
        pause_ratio = self._mean([item["pause_ratio"] for item in values], 0.0)
        repeat_similarity = self._mean([item["repeat_similarity"] for item in values], 0.0)
        envelope_jitter = self._mean([item["envelope_jitter"] for item in values], 0.0)
        duration_ratio_deviation = self._mean([item["duration_ratio_deviation"] for item in values], 0.0)
        acoustic_backend = str(values[0].get("acoustic_backend", "proxy"))
        acoustic_model_path = str(values[0].get("acoustic_model_path", ""))
        acoustic_model_compatibility = str(values[0].get("acoustic_model_compatibility", "unavailable"))
        acoustic_model_loaded = bool(values[0].get("acoustic_model_loaded", False))
        acoustic_model_error = str(values[0].get("acoustic_model_error", ""))
        return {
            "acoustic_embedding_distance": acoustic,
            "spectral_feature_distance": spectral,
            "formant_trajectory_distance": formant,
            "temporal_distance": temporal,
            "dtw_warp_distance": dtw,
            "duration_shape_distance": duration,
            "pause_ratio": pause_ratio,
            "repeat_similarity": repeat_similarity,
            "envelope_jitter": envelope_jitter,
            "duration_ratio_deviation": duration_ratio_deviation,
            "acoustic_backend": acoustic_backend,
            "acoustic_model_path": acoustic_model_path,
            "acoustic_model_compatibility": acoustic_model_compatibility,
            "acoustic_model_loaded": acoustic_model_loaded,
            "acoustic_model_error": acoustic_model_error,
        }

    def _extract_word_audio_features(
        self,
        audio: Optional[AudioBuffer],
        graph: PhoneGraph,
        skeletons: List[PronunciationSkeleton],
    ) -> Dict[int, Dict[str, Any]]:
        backend_status = self._get_acoustic_backend_status()
        if audio is None or not audio.samples:
            return self._fallback_timing_features(graph, skeletons, backend_status)

        features: Dict[int, Dict[str, Any]] = {}
        nodes_by_word: Dict[int, List[PhoneNode]] = {}
        for node in graph.nodes:
            nodes_by_word.setdefault(node.word_index, []).append(node)

        for skeleton in skeletons:
            nodes = nodes_by_word.get(skeleton.word_index, [])
            if not nodes:
                continue
            start_sec = min(node.start_sec or 0.0 for node in nodes)
            end_sec = max(node.end_sec or start_sec for node in nodes)
            segment = self._slice_audio(audio, start_sec, end_sec)
            phone_segments = [
                self._slice_audio(audio, node.start_sec or start_sec, node.end_sec or end_sec)
                for node in nodes
            ]
            frame_features = self._frame_features(segment)
            hubert_features = self._hubert_segment_features(
                segment=segment,
                phone_segments=phone_segments,
                sample_rate_hz=audio.sample_rate_hz,
                backend_status=backend_status,
            )
            contrast = self._phone_contrast(nodes, phone_segments)
            vowel_trajectory = self._vowel_trajectory(nodes, phone_segments)
            duration_shape = self._duration_shape_distance(nodes)
            pause_ratio = frame_features["pause_ratio"]
            repeat_similarity = frame_features["repeat_similarity"]
            envelope_jitter = frame_features["envelope_jitter"]
            duration_ratio_deviation = abs(frame_features["duration_ratio"] - 1.0)

            spectral_distance = max(0.0, min(1.0, 1.0 - contrast))
            formant_distance = max(0.0, min(1.0, 1.0 - vowel_trajectory))
            proxy_acoustic_distance = max(
                0.0,
                min(
                    1.0,
                    (spectral_distance * 0.45)
                    + (formant_distance * 0.25)
                    + (frame_features["spectral_flux"] * 0.15)
                    + (pause_ratio * 0.15),
                ),
            )
            if hubert_features["hubert_distance_available"] >= 1.0:
                acoustic_distance = max(
                    0.0,
                    min(
                        1.0,
                        (hubert_features["hubert_cosine_distance"] * 0.55)
                        + (spectral_distance * 0.25)
                        + (formant_distance * 0.12)
                        + (frame_features["spectral_flux"] * 0.08),
                    ),
                )
            else:
                acoustic_distance = proxy_acoustic_distance
            dtw_distance = max(
                0.0,
                min(1.0, (envelope_jitter * 0.45) + (pause_ratio * 0.35) + (repeat_similarity * 0.20)),
            )
            temporal_distance = max(
                0.0,
                min(1.0, (dtw_distance * 0.45) + (duration_shape * 0.35) + (pause_ratio * 0.20)),
            )

            features[skeleton.word_index] = {
                "acoustic_embedding_distance": round(acoustic_distance, 4),
                "spectral_feature_distance": round(spectral_distance, 4),
                "formant_trajectory_distance": round(formant_distance, 4),
                "temporal_distance": round(temporal_distance, 4),
                "dtw_warp_distance": round(dtw_distance, 4),
                "duration_shape_distance": round(duration_shape, 4),
                "pause_ratio": round(pause_ratio, 4),
                "repeat_similarity": round(repeat_similarity, 4),
                "envelope_jitter": round(envelope_jitter, 4),
                "duration_ratio_deviation": round(duration_ratio_deviation, 4),
                "proxy_acoustic_distance": round(proxy_acoustic_distance, 4),
                **hubert_features,
                **self._acoustic_backend_metadata(backend_status),
            }
        return features

    def _fallback_timing_features(
        self,
        graph: PhoneGraph,
        skeletons: List[PronunciationSkeleton],
        backend_status: Optional[AcousticBackendStatus] = None,
    ) -> Dict[int, Dict[str, Any]]:
        features: Dict[int, Dict[str, Any]] = {}
        by_word: Dict[int, List[PhoneNode]] = {}
        for node in graph.nodes:
            by_word.setdefault(node.word_index, []).append(node)
        for skeleton in skeletons:
            nodes = by_word.get(skeleton.word_index, [])
            duration_shape = self._duration_shape_distance(nodes)
            features[skeleton.word_index] = {
                "acoustic_embedding_distance": 0.0,
                "spectral_feature_distance": 0.0,
                "formant_trajectory_distance": 0.0,
                "temporal_distance": round(duration_shape * 0.7, 4),
                "dtw_warp_distance": round(duration_shape * 0.65, 4),
                "duration_shape_distance": round(duration_shape, 4),
                "pause_ratio": 0.0,
                "repeat_similarity": 0.0,
                "envelope_jitter": round(duration_shape * 0.5, 4),
                "duration_ratio_deviation": round(duration_shape, 4),
                "proxy_acoustic_distance": 0.0,
                "hubert_distance_available": 0.0,
                "hubert_cosine_distance": 0.0,
                "hubert_frame_count": 0.0,
                "hubert_phone_embedding_count": 0.0,
                **self._acoustic_backend_metadata(backend_status),
            }
        return features

    def _get_acoustic_backend_status(self) -> AcousticBackendStatus:
        if self._acoustic_backend_status is not None:
            return self._acoustic_backend_status

        model_path = str(self._default_hubert_path)
        try:
            import torch
            import torchaudio
        except Exception as exc:
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="torchaudio_unavailable",
                loaded=False,
                load_error=str(exc),
            )
            return self._acoustic_backend_status

        if not self._default_hubert_path.exists():
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="checkpoint_missing",
                loaded=False,
                load_error="checkpoint_missing",
                torch_module=torch,
                torchaudio_module=torchaudio,
            )
            return self._acoustic_backend_status

        try:
            checkpoint = torch.load(
                str(self._default_hubert_path),
                map_location="cpu",
                weights_only=False,
            )
        except ModuleNotFoundError as exc:
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="fairseq_checkpoint_incompatible",
                loaded=False,
                load_error=str(exc),
                torch_module=torch,
                torchaudio_module=torchaudio,
            )
            return self._acoustic_backend_status
        except Exception as exc:
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="checkpoint_load_failed",
                loaded=False,
                load_error=str(exc),
                torch_module=torch,
                torchaudio_module=torchaudio,
            )
            return self._acoustic_backend_status

        if isinstance(checkpoint, dict) and any(key in checkpoint for key in ("cfg", "args", "task_state")):
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="fairseq_checkpoint_incompatible",
                loaded=False,
                load_error="fairseq_checkpoint_requires_fairseq_runtime",
                torch_module=torch,
                torchaudio_module=torchaudio,
            )
            return self._acoustic_backend_status

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="unsupported_checkpoint_payload",
                loaded=False,
                load_error="unsupported_checkpoint_payload",
                torch_module=torch,
                torchaudio_module=torchaudio,
            )
            return self._acoustic_backend_status

        try:
            model = torchaudio.models.hubert_base()
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        except Exception as exc:
            self._acoustic_backend_status = AcousticBackendStatus(
                backend_name="proxy",
                model_path=model_path,
                compatibility="torchaudio_state_dict_incompatible",
                loaded=False,
                load_error=str(exc),
                torch_module=torch,
                torchaudio_module=torchaudio,
            )
            return self._acoustic_backend_status

        self._acoustic_backend_status = AcousticBackendStatus(
            backend_name="torchaudio_hubert",
            model_path=model_path,
            compatibility="torchaudio_state_dict_loaded",
            loaded=True,
            model=model,
            torch_module=torch,
            torchaudio_module=torchaudio,
        )
        return self._acoustic_backend_status

    def _hubert_segment_features(
        self,
        segment: List[float],
        phone_segments: List[List[float]],
        sample_rate_hz: int,
        backend_status: AcousticBackendStatus,
    ) -> Dict[str, float | str | bool]:
        if not backend_status.loaded or backend_status.model is None or backend_status.torch_module is None:
            return {
                "hubert_distance_available": 0.0,
                "hubert_cosine_distance": 0.0,
                "hubert_frame_count": 0.0,
                "hubert_phone_embedding_count": 0.0,
            }

        word_embedding = self._hubert_embedding(segment, sample_rate_hz, backend_status)
        if word_embedding is None:
            return {
                "hubert_distance_available": 0.0,
                "hubert_cosine_distance": 0.0,
                "hubert_frame_count": 0.0,
                "hubert_phone_embedding_count": 0.0,
            }

        phone_embeddings = [
            embedding
            for embedding in (
                self._hubert_embedding(phone_segment, sample_rate_hz, backend_status)
                for phone_segment in phone_segments
            )
            if embedding is not None
        ]
        if not phone_embeddings:
            return {
                "hubert_distance_available": 0.0,
                "hubert_cosine_distance": 0.0,
                "hubert_frame_count": float(word_embedding.shape[0]),
                "hubert_phone_embedding_count": 0.0,
            }

        distances = [
            self._cosine_distance(word_embedding, phone_embedding, backend_status.torch_module)
            for phone_embedding in phone_embeddings
        ]
        return {
            "hubert_distance_available": 1.0,
            "hubert_cosine_distance": round(self._mean(distances, 0.0), 4),
            "hubert_frame_count": float(word_embedding.shape[0]),
            "hubert_phone_embedding_count": float(len(phone_embeddings)),
        }

    def _hubert_embedding(
        self,
        samples: List[float],
        sample_rate_hz: int,
        backend_status: AcousticBackendStatus,
    ) -> Any:
        if (
            not backend_status.loaded
            or backend_status.model is None
            or backend_status.torch_module is None
            or backend_status.torchaudio_module is None
        ):
            return None
        torch = backend_status.torch_module
        torchaudio = backend_status.torchaudio_module
        if not samples:
            return None
        waveform = torch.tensor(samples, dtype=torch.float32)
        if waveform.numel() < 400:
            waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.numel()))
        if sample_rate_hz != 16000:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate_hz,
                16000,
            ).squeeze(0)
        waveform = waveform.unsqueeze(0)
        try:
            with torch.no_grad():
                features, _ = backend_status.model.extract_features(waveform)
        except Exception:
            return None
        if not features:
            return None
        return features[-1].mean(dim=1).squeeze(0)

    def _cosine_distance(self, left: Any, right: Any, torch_module: object) -> float:
        similarity = torch_module.nn.functional.cosine_similarity(left, right, dim=0)
        value = float(similarity.item())
        return max(0.0, min(1.0, 1.0 - ((value + 1.0) / 2.0)))

    def _acoustic_backend_metadata(
        self,
        backend_status: Optional[AcousticBackendStatus] = None,
    ) -> Dict[str, str | bool]:
        status = backend_status or self._get_acoustic_backend_status()
        return {
            "acoustic_backend": status.backend_name,
            "acoustic_model_path": status.model_path,
            "acoustic_model_compatibility": status.compatibility,
            "acoustic_model_loaded": status.loaded,
            "acoustic_model_error": status.load_error or "",
        }

    def _acoustic_signal_sources(self, audio_features: Dict[str, Any]) -> List[str]:
        sources = ["audio_window"]
        if float(audio_features.get("hubert_distance_available", 0.0)) >= 1.0:
            sources.append("torchaudio_hubert")
        else:
            sources.append("proxy_acoustic_features")
        return sources

    def _slice_audio(self, audio: AudioBuffer, start_sec: float, end_sec: float) -> List[float]:
        start = max(0, min(len(audio.samples), int(round(start_sec * audio.sample_rate_hz))))
        end = max(start + 1, min(len(audio.samples), int(round(end_sec * audio.sample_rate_hz))))
        return list(audio.samples[start:end])

    def _frame_features(self, segment: List[float]) -> Dict[str, float]:
        if not segment:
            return {
                "pause_ratio": 0.0,
                "repeat_similarity": 0.0,
                "envelope_jitter": 0.0,
                "spectral_flux": 0.0,
                "duration_ratio": 1.0,
            }
        frame_size = max(16, min(256, len(segment) // 8 or 16))
        frames = [segment[index : index + frame_size] for index in range(0, len(segment), frame_size)]
        rms_values = [self._rms(frame) for frame in frames if frame]
        diff_values = [self._mean_abs_diff(frame) for frame in frames if frame]
        max_rms = max(rms_values) if rms_values else 0.0
        pause_threshold = max_rms * 0.18
        pause_ratio = (
            sum(1 for value in rms_values if value <= pause_threshold) / float(len(rms_values) or 1)
        )
        envelope_jitter = self._normalized_mean_delta(rms_values)
        spectral_flux = self._normalized_mean_delta(diff_values)
        repeat_similarity = self._adjacent_similarity(frames)
        duration_ratio = max(0.5, min(1.8, len(segment) / float(max(frame_size * 8, 1))))
        return {
            "pause_ratio": max(0.0, min(1.0, pause_ratio)),
            "repeat_similarity": max(0.0, min(1.0, repeat_similarity)),
            "envelope_jitter": max(0.0, min(1.0, envelope_jitter)),
            "spectral_flux": max(0.0, min(1.0, spectral_flux)),
            "duration_ratio": duration_ratio,
        }

    def _phone_contrast(self, nodes: List[PhoneNode], phone_segments: List[List[float]]) -> float:
        if not nodes or not phone_segments:
            return 1.0
        consonant_energies = []
        vowel_energies = []
        for node, segment in zip(nodes, phone_segments):
            energy = self._mean_abs_diff(segment)
            if self._is_vowel(node.phone):
                vowel_energies.append(energy)
            else:
                consonant_energies.append(energy)
        consonant_mean = self._mean(consonant_energies, 0.0)
        vowel_mean = self._mean(vowel_energies, 0.0)
        if consonant_mean <= 1e-6 and vowel_mean <= 1e-6:
            return 1.0
        return max(0.0, min(1.0, abs(consonant_mean - vowel_mean) / max(consonant_mean, vowel_mean, 1e-6)))

    def _vowel_trajectory(self, nodes: List[PhoneNode], phone_segments: List[List[float]]) -> float:
        trajectories: List[float] = []
        for node, segment in zip(nodes, phone_segments):
            if not self._is_vowel(node.phone) or len(segment) < 4:
                continue
            midpoint = len(segment) // 2
            first = self._mean_abs_diff(segment[:midpoint])
            second = self._mean_abs_diff(segment[midpoint:])
            trajectories.append(abs(second - first))
        if not trajectories:
            return 1.0
        return max(0.0, min(1.0, self._mean(trajectories, 0.0) * 8.0))

    def _duration_shape_distance(self, nodes: List[PhoneNode]) -> float:
        durations = [
            max(0.0, float((node.end_sec or 0.0) - (node.start_sec or 0.0)))
            for node in nodes
            if node.start_sec is not None and node.end_sec is not None
        ]
        if len(durations) <= 1:
            return 0.0
        mean_duration = self._mean(durations, 0.0)
        if mean_duration <= 1e-6:
            return 0.0
        variance = sum((duration - mean_duration) ** 2 for duration in durations) / float(len(durations))
        coeff = math.sqrt(variance) / mean_duration
        return round(max(0.0, min(1.0, coeff / 0.55)), 4)

    def _confidence_for_evidence(self, values: List[float]) -> float:
        if not values:
            return 0.55
        if len(values) == 1:
            return 0.62
        mean_value = self._mean(values, 0.0)
        variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
        spread = math.sqrt(variance)
        confidence = 0.9 - min(0.35, spread)
        return round(max(0.5, min(0.95, confidence)), 4)

    def _mean(self, values: List[float], default: float) -> float:
        if not values:
            return default
        return sum(values) / float(len(values))

    def _rms(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return math.sqrt(sum(value * value for value in values) / float(len(values)))

    def _mean_abs_diff(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        diffs = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
        return sum(diffs) / float(len(diffs))

    def _normalized_mean_delta(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        diffs = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
        scale = max(max(values), 1e-6)
        return min(1.0, (sum(diffs) / float(len(diffs))) / scale)

    def _adjacent_similarity(self, frames: List[List[float]]) -> float:
        if len(frames) <= 1:
            return 0.0
        similarities = []
        for left, right in zip(frames, frames[1:]):
            limit = min(len(left), len(right))
            if limit <= 0:
                continue
            diff = sum(abs(left[index] - right[index]) for index in range(limit)) / float(limit)
            baseline = max(
                sum(abs(value) for value in left[:limit]) / float(limit),
                sum(abs(value) for value in right[:limit]) / float(limit),
                1e-6,
            )
            similarities.append(max(0.0, min(1.0, 1.0 - (diff / baseline))))
        return self._mean(similarities, 0.0)

    def _phone_order_similarity(self, left: List[str], right: List[str]) -> float:
        if not left:
            return 1.0
        matches = sum(
            1 for index in range(min(len(left), len(right))) if left[index] == right[index]
        )
        return matches / float(len(left))

    def _rime(self, value: str) -> str:
        vowels = "aeiouy"
        for index, char in enumerate(value):
            if char in vowels:
                return value[index:]
        return value

    def _feature_bundle(self, phone: str) -> PhoneFeatureBundle:
        consonants = {
            "P": {"place": "bilabial", "manner": "stop", "voicing": "voiceless"},
            "B": {"place": "bilabial", "manner": "stop", "voicing": "voiced"},
            "T": {"place": "alveolar", "manner": "stop", "voicing": "voiceless"},
            "D": {"place": "alveolar", "manner": "stop", "voicing": "voiced"},
            "K": {"place": "velar", "manner": "stop", "voicing": "voiceless"},
            "G": {"place": "velar", "manner": "stop", "voicing": "voiced"},
            "M": {"place": "bilabial", "manner": "nasal", "voicing": "voiced"},
            "N": {"place": "alveolar", "manner": "nasal", "voicing": "voiced"},
            "NG": {"place": "velar", "manner": "nasal", "voicing": "voiced"},
            "F": {"place": "labiodental", "manner": "fricative", "voicing": "voiceless"},
            "V": {"place": "labiodental", "manner": "fricative", "voicing": "voiced"},
            "S": {"place": "alveolar", "manner": "fricative", "voicing": "voiceless"},
            "Z": {"place": "alveolar", "manner": "fricative", "voicing": "voiced"},
            "SH": {"place": "postalveolar", "manner": "fricative", "voicing": "voiceless"},
            "TH": {"place": "dental", "manner": "fricative", "voicing": "voiceless"},
            "DH": {"place": "dental", "manner": "fricative", "voicing": "voiced"},
            "CH": {"place": "postalveolar", "manner": "affricate", "voicing": "voiceless"},
            "JH": {"place": "postalveolar", "manner": "affricate", "voicing": "voiced"},
            "L": {"place": "alveolar", "manner": "liquid", "voicing": "voiced"},
            "R": {"place": "postalveolar", "manner": "liquid", "voicing": "voiced"},
            "W": {"place": "labiovelar", "manner": "glide", "voicing": "voiced"},
            "Y": {"place": "palatal", "manner": "glide", "voicing": "voiced"},
            "HH": {"place": "glottal", "manner": "fricative", "voicing": "voiceless"},
        }
        vowels = {
            "AE": {"vowel_height": "low", "vowel_backness": "front", "rounding": "unrounded"},
            "AH": {"vowel_height": "mid", "vowel_backness": "central", "rounding": "unrounded"},
            "EH": {"vowel_height": "mid", "vowel_backness": "front", "rounding": "unrounded"},
            "IH": {"vowel_height": "high", "vowel_backness": "front", "rounding": "unrounded"},
            "IY": {"vowel_height": "high", "vowel_backness": "front", "rounding": "unrounded"},
            "OW": {"vowel_height": "mid", "vowel_backness": "back", "rounding": "rounded"},
            "UW": {"vowel_height": "high", "vowel_backness": "back", "rounding": "rounded"},
            "AY": {"vowel_height": "diphthong", "vowel_backness": "central", "rounding": "unrounded"},
            "EY": {"vowel_height": "diphthong", "vowel_backness": "front", "rounding": "unrounded"},
            "OY": {"vowel_height": "diphthong", "vowel_backness": "back", "rounding": "rounded"},
            "AW": {"vowel_height": "diphthong", "vowel_backness": "central", "rounding": "rounded"},
            "ER": {"vowel_height": "mid", "vowel_backness": "central", "rounding": "unrounded"},
            "AA": {"vowel_height": "low", "vowel_backness": "back", "rounding": "unrounded"},
        }
        payload = consonants.get(phone, vowels.get(phone, {}))
        return PhoneFeatureBundle(
            place=payload.get("place"),
            manner=payload.get("manner"),
            voicing=payload.get("voicing"),
            vowel_height=payload.get("vowel_height"),
            vowel_backness=payload.get("vowel_backness"),
            rounding=payload.get("rounding"),
        )

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER", "AA"}
