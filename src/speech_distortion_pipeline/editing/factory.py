from __future__ import annotations

from .source_editor import (
    HeuristicPronunciationSourceEditor,
    HeuristicSourceSegmentEditor,
    PronunciationSourceEditor,
    SourceSegmentEditor,
)


def build_source_editor() -> SourceSegmentEditor:
    return HeuristicSourceSegmentEditor()


def build_pronunciation_source_editor() -> PronunciationSourceEditor:
    return HeuristicPronunciationSourceEditor()
