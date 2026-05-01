from __future__ import annotations

from speech_distortion_pipeline.models import SeverityProfile

from .planner import ErrorPlanner, HeuristicErrorPlanner


def build_severity_profile(
    global_severity: float,
    complexity_slope: float,
    distortion_bias: float,
    substitution_bias: float,
    addition_bias: float,
    vowel_lengthening_bias: float,
    consonant_lengthening_bias: float,
) -> SeverityProfile:
    return SeverityProfile(
        global_severity=global_severity,
        complexity_slope=complexity_slope,
        distortion_bias=distortion_bias,
        substitution_bias=substitution_bias,
        addition_bias=addition_bias,
        vowel_lengthening_bias=vowel_lengthening_bias,
        consonant_lengthening_bias=consonant_lengthening_bias,
    )


def build_severity_profile_from_config(config: object) -> SeverityProfile:
    return build_severity_profile(
        global_severity=float(config.global_severity),
        complexity_slope=float(config.complexity_slope),
        distortion_bias=float(config.distortion_bias),
        substitution_bias=float(config.substitution_bias),
        addition_bias=float(config.addition_bias),
        vowel_lengthening_bias=float(config.vowel_lengthening_bias),
        consonant_lengthening_bias=float(config.consonant_lengthening_bias),
    )


def build_error_planner() -> ErrorPlanner:
    return HeuristicErrorPlanner()
