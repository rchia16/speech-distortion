from __future__ import annotations

from speech_distortion_pipeline.config import HybridConfig

from .budget import (
    DurationBudgetManager,
    HeuristicDurationBudgetManager,
    HeuristicPronunciationTimingPlanner,
    PronunciationTimingPlanner,
)


def build_duration_budget_manager() -> DurationBudgetManager:
    return HeuristicDurationBudgetManager()


def build_hybrid_timing_planner(config: HybridConfig) -> PronunciationTimingPlanner:
    return HeuristicPronunciationTimingPlanner(config=config)
