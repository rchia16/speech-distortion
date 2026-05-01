from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from speech_distortion_pipeline.models import PhoneGraph, WordComplexity


class ComplexityScorer(Protocol):
    """Assigns word-level articulatory complexity used by the planner."""

    def score(self, graph: PhoneGraph) -> PhoneGraph:
        raise NotImplementedError


@dataclass
class HeuristicComplexityScorer:
    """Word-level complexity scoring used to scale later distortions."""

    marked_clusters: tuple[str, ...] = ("STR", "SPR", "SKR", "SPL", "THR", "SKW")

    def score(self, graph: PhoneGraph) -> PhoneGraph:
        grouped: Dict[int, List] = {}
        order: List[tuple[int, str]] = []
        for node in graph.nodes:
            if node.word_index not in grouped:
                order.append((node.word_index, node.word))
            grouped.setdefault(node.word_index, []).append(node)

        graph.word_complexities = []
        for index, item in enumerate(order):
            word_index, word = item
            nodes = grouped[word_index]
            consonant_clusters = self._cluster_lengths(nodes)
            cluster_score = sum(max(length - 1, 0) * 0.35 for length in consonant_clusters)

            phonotactic_score = 0.0
            phone_string = "".join(node.phone for node in nodes)
            for marked in self.marked_clusters:
                if marked in phone_string:
                    phonotactic_score += 0.6

            later_count = sum(1 for node in nodes if node.features.later_developing)
            later_developing_score = later_count / float(max(len(nodes), 1))

            syllable_count = max(1, len([node for node in nodes if self._is_vowel(node.phone)]))
            syllable_score = max(0, syllable_count - 1) * 0.25

            score = round(cluster_score + phonotactic_score + later_developing_score + syllable_score, 4)
            graph.word_complexities.append(
                WordComplexity(
                    word=word,
                    index=index,
                    score=score,
                    cluster_score=round(cluster_score, 4),
                    phonotactic_score=round(phonotactic_score, 4),
                    later_developing_score=round(later_developing_score, 4),
                    syllable_score=round(syllable_score, 4),
                )
            )
        return graph

    def _cluster_lengths(self, nodes: List) -> List[int]:
        lengths: List[int] = []
        current = 0
        for node in nodes:
            if self._is_vowel(node.phone):
                if current > 1:
                    lengths.append(current)
                current = 0
            else:
                current += 1
        if current > 1:
            lengths.append(current)
        return lengths

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER"}
