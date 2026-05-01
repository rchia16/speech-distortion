from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from speech_distortion_pipeline.models import AlignmentResult, PhoneFeatureBundle, PhoneGraph, PhoneNode

from .g2p import GraphemeToPhoneme, HeuristicEnglishG2P


class FeatureExtractor(Protocol):
    """Attaches articulatory and prosodic features to aligned phones."""

    def build_phone_graph(self, alignment: AlignmentResult) -> PhoneGraph:
        raise NotImplementedError


@dataclass
class HeuristicFeatureExtractor:
    g2p: GraphemeToPhoneme

    _FEATURES: Dict[str, Dict[str, object]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._FEATURES is None:
            self._FEATURES = {
                "P": {"place": "bilabial", "manner": "stop", "voicing": "voiceless", "sonority": 1.0},
                "B": {"place": "bilabial", "manner": "stop", "voicing": "voiced", "sonority": 1.0},
                "T": {"place": "alveolar", "manner": "stop", "voicing": "voiceless", "sonority": 1.0},
                "D": {"place": "alveolar", "manner": "stop", "voicing": "voiced", "sonority": 1.0},
                "K": {"place": "velar", "manner": "stop", "voicing": "voiceless", "sonority": 1.0},
                "G": {"place": "velar", "manner": "stop", "voicing": "voiced", "sonority": 1.0},
                "M": {"place": "bilabial", "manner": "nasal", "voicing": "voiced", "sonority": 3.0},
                "N": {"place": "alveolar", "manner": "nasal", "voicing": "voiced", "sonority": 3.0},
                "NG": {"place": "velar", "manner": "nasal", "voicing": "voiced", "sonority": 3.0},
                "F": {"place": "labiodental", "manner": "fricative", "voicing": "voiceless", "sonority": 2.0},
                "V": {"place": "labiodental", "manner": "fricative", "voicing": "voiced", "sonority": 2.0},
                "S": {"place": "alveolar", "manner": "fricative", "voicing": "voiceless", "sonority": 2.0},
                "Z": {"place": "alveolar", "manner": "fricative", "voicing": "voiced", "sonority": 2.0},
                "SH": {"place": "postalveolar", "manner": "fricative", "voicing": "voiceless", "sonority": 2.0},
                "TH": {"place": "dental", "manner": "fricative", "voicing": "voiceless", "sonority": 2.0},
                "DH": {"place": "dental", "manner": "fricative", "voicing": "voiced", "sonority": 2.0},
                "HH": {"place": "glottal", "manner": "fricative", "voicing": "voiceless", "sonority": 2.0},
                "CH": {"place": "postalveolar", "manner": "affricate", "voicing": "voiceless", "sonority": 1.5},
                "JH": {"place": "postalveolar", "manner": "affricate", "voicing": "voiced", "sonority": 1.5},
                "L": {"place": "alveolar", "manner": "liquid", "voicing": "voiced", "sonority": 4.0},
                "R": {"place": "postalveolar", "manner": "liquid", "voicing": "voiced", "sonority": 4.0},
                "W": {"place": "labiovelar", "manner": "glide", "voicing": "voiced", "sonority": 4.5},
                "Y": {"place": "palatal", "manner": "glide", "voicing": "voiced", "sonority": 4.5},
                "AE": {"vowel_height": "low", "vowel_backness": "front", "rounding": "unrounded", "sonority": 5.0},
                "AH": {"vowel_height": "mid", "vowel_backness": "central", "rounding": "unrounded", "sonority": 5.0},
                "EH": {"vowel_height": "mid", "vowel_backness": "front", "rounding": "unrounded", "sonority": 5.0},
                "IH": {"vowel_height": "high", "vowel_backness": "front", "rounding": "unrounded", "sonority": 5.0},
                "IY": {"vowel_height": "high", "vowel_backness": "front", "rounding": "unrounded", "sonority": 5.0},
                "OW": {"vowel_height": "mid", "vowel_backness": "back", "rounding": "rounded", "sonority": 5.0},
                "UW": {"vowel_height": "high", "vowel_backness": "back", "rounding": "rounded", "sonority": 5.0},
                "AY": {"vowel_height": "diphthong", "vowel_backness": "central", "rounding": "unrounded", "sonority": 5.0},
                "EY": {"vowel_height": "diphthong", "vowel_backness": "front", "rounding": "unrounded", "sonority": 5.0},
                "OY": {"vowel_height": "diphthong", "vowel_backness": "back", "rounding": "rounded", "sonority": 5.0},
                "AW": {"vowel_height": "diphthong", "vowel_backness": "central", "rounding": "rounded", "sonority": 5.0},
                "ER": {"vowel_height": "mid", "vowel_backness": "central", "rounding": "unrounded", "sonority": 5.0},
            }

    def build_phone_graph(self, alignment: AlignmentResult) -> PhoneGraph:
        nodes: List[PhoneNode] = []
        node_index = 0
        for word in alignment.words:
            word_phones = self._phonemize_word(word.text)
            if not word_phones:
                continue

            duration = max(word.end_sec - word.start_sec, 0.0)
            phone_duration = duration / float(len(word_phones)) if word_phones else 0.0
            syllable_index = 0
            seen_vowel = False
            for phone_offset, phone in enumerate(word_phones):
                start_sec = word.start_sec + (phone_offset * phone_duration)
                end_sec = word.end_sec if phone_offset == len(word_phones) - 1 else start_sec + phone_duration

                if self._is_vowel(phone):
                    if seen_vowel:
                        syllable_index += 1
                    seen_vowel = True

                nodes.append(
                    PhoneNode(
                        index=node_index,
                        phone=phone,
                        word=word.text,
                        word_index=word.index,
                        features=self._feature_bundle(phone),
                        stress="primary" if self._is_primary_stress(phone_offset, word_phones) else None,
                        syllable_index=syllable_index,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
                )
                node_index += 1

        self._annotate_clusters(nodes)
        return PhoneGraph(transcript=alignment.transcript, nodes=nodes)

    def _phonemize_word(self, word: str) -> List[str]:
        if isinstance(self.g2p, HeuristicEnglishG2P):
            return self.g2p.phonemize_word(word)
        return self.g2p.phonemize(word)

    def _feature_bundle(self, phone: str) -> PhoneFeatureBundle:
        payload = self._FEATURES.get(phone, {})
        later_developing = phone in {"R", "L", "S", "Z", "SH", "CH", "JH", "TH", "DH"}
        return PhoneFeatureBundle(
            place=payload.get("place"),
            manner=payload.get("manner"),
            voicing=payload.get("voicing"),
            vowel_height=payload.get("vowel_height"),
            vowel_backness=payload.get("vowel_backness"),
            rounding=payload.get("rounding"),
            sonority=payload.get("sonority"),
            later_developing=later_developing,
        )

    def _annotate_clusters(self, nodes: List[PhoneNode]) -> None:
        by_word: Dict[int, List[PhoneNode]] = {}
        for node in nodes:
            by_word.setdefault(node.word_index, []).append(node)

        for word_nodes in by_word.values():
            run: List[PhoneNode] = []
            for node in word_nodes + [None]:  # type: ignore[list-item]
                if node is not None and not self._is_vowel(node.phone):
                    run.append(node)
                    continue
                if len(run) > 1:
                    for item in run:
                        item.features.cluster_member = True
                        item.features.cluster_size = len(run)
                run = []

    def _is_primary_stress(self, phone_offset: int, phones: List[str]) -> bool:
        for index, phone in enumerate(phones):
            if self._is_vowel(phone):
                return index == phone_offset
        return phone_offset == 0

    def _is_vowel(self, phone: str) -> bool:
        return phone in {"AE", "AH", "EH", "IH", "IY", "OW", "UW", "AY", "EY", "OY", "AW", "ER"}
