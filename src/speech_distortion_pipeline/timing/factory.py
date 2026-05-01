from __future__ import annotations

from .budget import DurationBudgetManager, HeuristicDurationBudgetManager


def build_duration_budget_manager() -> DurationBudgetManager:
    return HeuristicDurationBudgetManager()
