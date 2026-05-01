from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhoneSpan:
    phone: str
    start_sec: float
    end_sec: float
    word_index: int
    syllable_index: Optional[int] = None
    stress: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class WordSpan:
    text: str
    start_sec: float
    end_sec: float
    index: int
    phones: list[PhoneSpan] = field(default_factory=list)


@dataclass
class AlignmentResult:
    transcript: str
    words: list[WordSpan]
    phones: list[PhoneSpan]
    backend_name: str
