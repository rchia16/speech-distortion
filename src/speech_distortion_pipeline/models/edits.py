from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EditType(str, Enum):
    KEEP = "keep"
    SUBSTITUTE = "substitute"
    DISTORTED_SUBSTITUTE = "distorted_substitute"
    ADD = "add"
    DISTORT = "distort"
    LENGTHEN_VOWEL = "lengthen_vowel"
    LENGTHEN_CONSONANT = "lengthen_consonant"
    COMPENSATORY_SHORTEN = "compensatory_shorten"


@dataclass
class SeverityProfile:
    global_severity: float
    complexity_slope: float
    distortion_bias: float
    substitution_bias: float
    addition_bias: float
    vowel_lengthening_bias: float
    consonant_lengthening_bias: float


@dataclass
class EditOperation:
    edit_type: EditType
    target_phone_indices: list[int]
    replacement_phones: list[str] = field(default_factory=list)
    inserted_phones: list[str] = field(default_factory=list)
    target_duration_delta_sec: float = 0.0
    complexity_weight: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class EditPlan:
    operations: list[EditOperation]
    planner_name: str
