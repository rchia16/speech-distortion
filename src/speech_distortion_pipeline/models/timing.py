from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DurationBudget:
    window_start_phone_index: int
    window_end_phone_index: int
    requested_delta_sec: float
    compensated_delta_sec: float
    preserved_total_duration: bool = True


@dataclass
class TimingPlan:
    budgets: list[DurationBudget] = field(default_factory=list)
    total_requested_delta_sec: float = 0.0
    total_compensated_delta_sec: float = 0.0
