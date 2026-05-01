from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhoneFeatureBundle:
    place: Optional[str] = None
    manner: Optional[str] = None
    voicing: Optional[str] = None
    vowel_height: Optional[str] = None
    vowel_backness: Optional[str] = None
    rounding: Optional[str] = None
    sonority: Optional[float] = None
    later_developing: bool = False
    cluster_member: bool = False
    cluster_size: int = 0


@dataclass
class PhoneNode:
    index: int
    phone: str
    word: str
    word_index: int
    features: PhoneFeatureBundle
    stress: Optional[str] = None
    syllable_index: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


@dataclass
class WordComplexity:
    word: str
    index: int
    score: float
    cluster_score: float
    phonotactic_score: float
    later_developing_score: float
    syllable_score: float


@dataclass
class PhoneGraph:
    transcript: str
    nodes: list[PhoneNode]
    word_complexities: list[WordComplexity] = field(default_factory=list)
