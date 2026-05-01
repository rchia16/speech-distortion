from __future__ import annotations

from .projector import HeuristicTimbreProjector, TimbreProjector


def build_timbre_projector() -> TimbreProjector:
    return HeuristicTimbreProjector()
