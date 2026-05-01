from __future__ import annotations

from .source_editor import HeuristicSourceSegmentEditor, SourceSegmentEditor


def build_source_editor() -> SourceSegmentEditor:
    return HeuristicSourceSegmentEditor()
