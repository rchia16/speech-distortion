from __future__ import annotations

from speech_distortion_pipeline.config import PronunciationSliderConfig

from .budget import (
    DurationBudgetManager,
    HeuristicDurationBudgetManager,
    HeuristicPronunciationTimingPlanner,
    PronunciationTimingPlanner,
)


def build_duration_budget_manager() -> DurationBudgetManager:
    return HeuristicDurationBudgetManager()


def build_pronunciation_timing_planner(config: PronunciationSliderConfig) -> PronunciationTimingPlanner:
    return HeuristicPronunciationTimingPlanner(config=config)
