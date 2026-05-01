from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schema import (
    AlignmentConfig,
    EditingConfig,
    PhonologyConfig,
    PipelineConfig,
    ResynthesisConfig,
    SeverityConfig,
    StitchingConfig,
    TimbreConfig,
)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_yaml_subset(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError("Invalid config line: {!r}".format(raw_line))

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if raw_value == "":
            child: Dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(raw_value)

    return root


def _require_mapping(value: Any, section_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Expected '{}' to be a mapping.".format(section_name))
    return value


def load_config(path: Any) -> PipelineConfig:
    raw = _load_yaml_subset(Path(path))
    root = _require_mapping(raw, "root")

    pipeline = _require_mapping(root.get("pipeline"), "pipeline")
    severity = _require_mapping(root.get("severity"), "severity")
    alignment = _require_mapping(root.get("alignment"), "alignment")
    phonology = _require_mapping(root.get("phonology"), "phonology")
    editing = _require_mapping(root.get("editing"), "editing")
    resynthesis = _require_mapping(root.get("resynthesis"), "resynthesis")
    timbre = _require_mapping(root.get("timbre"), "timbre")
    stitching = _require_mapping(root.get("stitching"), "stitching")

    return PipelineConfig(
        sample_rate_hz=int(pipeline["sample_rate_hz"]),
        device=str(pipeline["device"]),
        preserve_total_duration=bool(pipeline["preserve_total_duration"]),
        target_latency_ms=int(pipeline["target_latency_ms"]),
        severity=SeverityConfig(
            global_severity=float(severity["global_severity"]),
            complexity_slope=float(severity["complexity_slope"]),
            distortion_bias=float(severity["distortion_bias"]),
            substitution_bias=float(severity["substitution_bias"]),
            addition_bias=float(severity["addition_bias"]),
            vowel_lengthening_bias=float(severity["vowel_lengthening_bias"]),
            consonant_lengthening_bias=float(severity["consonant_lengthening_bias"]),
        ),
        alignment=AlignmentConfig(
            runtime_backend=str(alignment["runtime_backend"]),
            offline_backend=str(alignment["offline_backend"]),
            emit_phone_confidence=bool(alignment["emit_phone_confidence"]),
        ),
        phonology=PhonologyConfig(
            g2p_backend=str(phonology["g2p_backend"]),
            language=str(phonology["language"]),
            compute_complexity=bool(phonology["compute_complexity"]),
        ),
        editing=EditingConfig(
            source_editor_backend=str(editing["source_editor_backend"]),
            enable_vowel_lengthening=bool(editing["enable_vowel_lengthening"]),
            enable_consonant_lengthening=bool(editing["enable_consonant_lengthening"]),
            enable_distortion_shaping=bool(editing["enable_distortion_shaping"]),
        ),
        resynthesis=ResynthesisConfig(
            backend=str(resynthesis["backend"]),
            use_context_phones=int(resynthesis["use_context_phones"]),
            max_fragment_ms=int(resynthesis["max_fragment_ms"]),
        ),
        timbre=TimbreConfig(
            backend=str(timbre["backend"]),
            enabled=bool(timbre["enabled"]),
        ),
        stitching=StitchingConfig(
            crossfade_ms=int(stitching["crossfade_ms"]),
            smooth_energy=bool(stitching["smooth_energy"]),
            smooth_f0=bool(stitching["smooth_f0"]),
        ),
    )
