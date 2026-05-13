from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import (
    AcceptanceTestConfig,
    AlignmentConfig,
    AnchorSelectionRulesConfig,
    EditingConfig,
    GlobalConstraintsConfig,
    OverrideModeConfig,
    PhonologyConfig,
    PipelineConfig,
    ProcessingStageConfig,
    PronunciationSimilarityScoreConfig,
    PronunciationSimilarityWeightsConfig,
    PronunciationSkeletonConfig,
    PronunciationSliderConfig,
    RangeConfig,
    ResynthesisConfig,
    SeverityConfig,
    ShortWordModeConfig,
    SliderConfig,
    SliderGuardrailsConfig,
    StitchingConfig,
    TimbreConfig,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level YAML mapping.")
    return payload


def _require_mapping(value: Any, section_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Expected '{}' to be a mapping.".format(section_name))
    return value


def load_config(path: Any) -> PipelineConfig:
    raw = _load_yaml(Path(path))
    pipeline = _require_mapping(raw.get("pipeline"), "pipeline")
    severity = _require_mapping(raw.get("severity"), "severity")
    alignment = _require_mapping(raw.get("alignment"), "alignment")
    phonology = _require_mapping(raw.get("phonology"), "phonology")
    editing = _require_mapping(raw.get("editing"), "editing")
    resynthesis = _require_mapping(raw.get("resynthesis"), "resynthesis")
    timbre = _require_mapping(raw.get("timbre"), "timbre")
    stitching = _require_mapping(raw.get("stitching"), "stitching")

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


def _parse_range(raw: Dict[str, Any]) -> RangeConfig:
    return RangeConfig(
        min=float(raw["min"]),
        max=float(raw["max"]),
        meaning=str(raw["meaning"]) if raw.get("meaning") is not None else None,
    )


def _parse_slider_guardrails(raw: Dict[str, Any]) -> SliderGuardrailsConfig:
    short_word_mode_raw = raw.get("short_word_mode")
    short_word_mode = None
    values = dict(raw)
    if isinstance(short_word_mode_raw, dict):
        short_word_mode = ShortWordModeConfig(
            enabled_when_phone_count_lte=int(short_word_mode_raw["enabled_when_phone_count_lte"]),
            max_phone_substitution_ratio=float(short_word_mode_raw["max_phone_substitution_ratio"]),
            max_deletions_per_word=int(short_word_mode_raw["max_deletions_per_word"]),
            preserve_first_phone=bool(short_word_mode_raw["preserve_first_phone"]),
            preserve_primary_vowel_nucleus=bool(short_word_mode_raw["preserve_primary_vowel_nucleus"]),
        )
        values.pop("short_word_mode", None)
    return SliderGuardrailsConfig(values=values, short_word_mode=short_word_mode)


def load_pronunciation_slider_config(path: Any) -> PronunciationSliderConfig:
    raw = _load_yaml(Path(path))

    global_constraints = _require_mapping(raw.get("global_constraints"), "global_constraints")
    skeleton = _require_mapping(raw.get("pronunciation_skeleton"), "pronunciation_skeleton")
    similarity = _require_mapping(raw.get("pronunciation_similarity_score"), "pronunciation_similarity_score")
    sliders_raw = _require_mapping(raw.get("sliders"), "sliders")

    slider_configs: Dict[str, SliderConfig] = {}
    for slider_name, slider_value in sliders_raw.items():
        slider_mapping = _require_mapping(slider_value, "sliders.{0}".format(slider_name))
        slider_configs[str(slider_name)] = SliderConfig(
            label=str(slider_mapping["label"]),
            default=float(slider_mapping["default"]),
            user_description=str(slider_mapping["user_description"]),
            independence_rule=str(slider_mapping["independence_rule"]),
            pronunciation_guardrails=_parse_slider_guardrails(
                _require_mapping(slider_mapping.get("pronunciation_guardrails"), "pronunciation_guardrails")
            ),
            generated_parameters=dict(_require_mapping(slider_mapping.get("generated_parameters"), "generated_parameters")),
            allowed_examples_for_no=[str(item) for item in slider_mapping.get("allowed_examples_for_no", [])],
            disallowed_examples_for_no=[str(item) for item in slider_mapping.get("disallowed_examples_for_no", [])],
        )

    processing_pipeline = [
        ProcessingStageConfig(
            id=str(item["id"]),
            outputs=[str(output) for output in item.get("outputs", [])],
            uses=[str(entry) for entry in item.get("uses", [])],
            condition=str(item["condition"]) if item.get("condition") is not None else None,
        )
        for item in raw.get("processing_pipeline", [])
    ]

    override_modes = {
        str(name): OverrideModeConfig(values=dict(_require_mapping(value, "override_modes.{0}".format(name))))
        for name, value in _require_mapping(raw.get("override_modes", {}), "override_modes").items()
    }

    acceptance_tests = [
        AcceptanceTestConfig(
            name=str(item["name"]),
            word=str(item["word"]),
            phones=[str(phone) for phone in item.get("phones", [])],
            controls={str(key): float(value) for key, value in _require_mapping(item.get("controls", {}), "controls").items()},
            expected=str(item["expected"]),
        )
        for item in raw.get("acceptance_tests", [])
    ]

    anchor_rules = _require_mapping(skeleton.get("anchor_selection_rules"), "anchor_selection_rules")
    weights = _require_mapping(similarity.get("weights"), "weights")

    return PronunciationSliderConfig(
        version=int(raw["version"]),
        kind=str(raw["kind"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        global_constraints=GlobalConstraintsConfig(
            slider_range=_parse_range(_require_mapping(global_constraints["slider_range"], "slider_range")),
            time_range=_parse_range(_require_mapping(global_constraints["time_range"], "time_range")),
            preserve_core_pronunciation=bool(global_constraints["preserve_core_pronunciation"]),
            allow_full_gibberish=bool(global_constraints["allow_full_gibberish"]),
            minimum_pronunciation_similarity=float(global_constraints["minimum_pronunciation_similarity"]),
            minimum_pronunciation_similarity_short_word=float(
                global_constraints["minimum_pronunciation_similarity_short_word"]
            ),
            max_total_duration_drift_ratio=float(global_constraints["max_total_duration_drift_ratio"]),
            random_seed=int(global_constraints["random_seed"]),
        ),
        pronunciation_skeleton=PronunciationSkeletonConfig(
            extract=[str(item) for item in skeleton.get("extract", [])],
            anchor_selection_rules=AnchorSelectionRulesConfig(
                always_anchor_primary_vowel=bool(anchor_rules["always_anchor_primary_vowel"]),
                always_anchor_first_content_consonant=bool(anchor_rules["always_anchor_first_content_consonant"]),
                anchor_final_consonant_for_closed_syllables=bool(anchor_rules["anchor_final_consonant_for_closed_syllables"]),
                short_word_anchor_all_phones_when_phone_count_lte=int(
                    anchor_rules["short_word_anchor_all_phones_when_phone_count_lte"]
                ),
            ),
        ),
        pronunciation_similarity_score=PronunciationSimilarityScoreConfig(
            range=[float(value) for value in similarity.get("range", [])],
            weights=PronunciationSimilarityWeightsConfig(
                anchor_phone_preservation=float(weights["anchor_phone_preservation"]),
                vowel_nucleus_similarity=float(weights["vowel_nucleus_similarity"]),
                syllable_count_similarity=float(weights["syllable_count_similarity"]),
                phone_order_similarity=float(weights["phone_order_similarity"]),
                duration_shape_similarity=float(weights["duration_shape_similarity"]),
            ),
            repair_order=[str(item) for item in similarity.get("repair_order", [])],
        ),
        sliders=slider_configs,
        processing_pipeline=processing_pipeline,
        override_modes=override_modes,
        acceptance_tests=acceptance_tests,
    )
