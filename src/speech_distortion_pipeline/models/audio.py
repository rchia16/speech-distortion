from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioBuffer:
    samples: List[float]
    sample_rate_hz: int
    channel_count: int = 1
    speaker_id: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AudioSegment:
    start_sec: float
    end_sec: float
    samples: Optional[List[float]] = None
    label: Optional[str] = None
    source: str = "original"
