from __future__ import annotations

import json
from pathlib import Path

from select_blabber_asset import (
    ASSET_MODE_GLOBAL,
    ASSET_MODE_PER_PHONEME,
    adapt_comparison_result,
    compute_per_phoneme_mismatch,
    load_asset_index,
    normalize_voice_mode,
    scan_asset_root,
    select_asset,
)


def test_normalize_voice_mode_accepts_external_aliases() -> None:
    assert normalize_voice_mode("female") == "woman"
    assert normalize_voice_mode("male") == "man"


def test_adapt_comparison_result_supports_v1_and_v2_shapes() -> None:
    v1 = adapt_comparison_result(
        {
            "label_id": 1,
            "label_name": "go",
            "times": [0.25, 0.75],
            "per_time_l2": [0.3, 0.6],
        }
    )
    v2 = adapt_comparison_result(
        {
            "label_id": 1,
            "label_name": "go",
            "times": [0.25, 0.75],
            "per_time_l2": [0.3, 0.6],
            "dtw_cost": 0.12,
            "dtw_path": [(0, 0), (1, 1)],
            "warped_input_seq": [[0.0], [1.0]],
        }
    )

    assert v1["source"] == "template_l2_compare"
    assert v2["source"] == "template_l2_compare_v2"
    assert v2["dtw_cost"] == 0.12


def test_compute_per_phoneme_mismatch_averages_by_alignment_span() -> None:
    scores = compute_per_phoneme_mismatch(
        times=[0.25, 0.75],
        per_time_l2=[0.5, 0.2],
        source_alignment=[
            {"phone": "G", "start_sec": 0.0, "end_sec": 0.5},
            {"phone": "OW", "start_sec": 0.5, "end_sec": 1.0},
        ],
    )

    assert [item["rounded_grade"] for item in scores] == [0.5, 0.2]


def _write_sidecar(path: Path, phones: list[str]) -> None:
    payload = {
        "source_alignment": [
            {
                "phone": phone,
                "start_sec": index / len(phones),
                "end_sec": (index + 1) / len(phones),
            }
            for index, phone in enumerate(phones)
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_asset_index_supports_nested_build_summary(tmp_path: Path) -> None:
    audio_path = tmp_path / "go_0.5_0.2.wav"
    sidecar_path = tmp_path / "go_0.5_0.2.json"
    audio_path.write_bytes(b"")
    _write_sidecar(sidecar_path, ["G", "OW"])

    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "transcript": "go",
                    "items": [
                        {
                            "audio_path": audio_path.name,
                            "json_path": sidecar_path.name,
                            "source_phonemes": ["G", "OW"],
                            "voice_mode": "woman",
                            "phoneme_values": [0.5, 0.2],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    candidates = load_asset_index(index_path)

    assert len(candidates) == 1
    assert candidates[0].word == "go"
    assert candidates[0].audio_path == audio_path.resolve()


def test_select_asset_prefers_exact_grade_match(tmp_path: Path) -> None:
    exact_audio = tmp_path / "go_exact.wav"
    exact_sidecar = tmp_path / "go_exact.json"
    near_audio = tmp_path / "go_near.wav"
    near_sidecar = tmp_path / "go_near.json"
    exact_audio.write_bytes(b"")
    near_audio.write_bytes(b"")
    _write_sidecar(exact_sidecar, ["G", "OW"])
    _write_sidecar(near_sidecar, ["G", "OW"])

    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "word": "go",
                    "audio_path": exact_audio.name,
                    "sidecar_path": exact_sidecar.name,
                    "source_phonemes": ["G", "OW"],
                    "voice_mode": "female",
                    "phoneme_values": [0.5, 0.2],
                },
                {
                    "word": "go",
                    "audio_path": near_audio.name,
                    "sidecar_path": near_sidecar.name,
                    "source_phonemes": ["G", "OW"],
                    "voice_mode": "female",
                    "phoneme_values": [0.5, 0.3],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = select_asset(
        comparison=adapt_comparison_result(
            {
                "label_name": "go",
                "times": [0.25, 0.75],
                "per_time_l2": [0.5, 0.2],
            }
        ),
        candidates=load_asset_index(index_path),
        requested_voice="woman",
    )

    assert result["exact_match"] is True
    assert result["audio_path"] == str(exact_audio.resolve())
    assert result["requested_phoneme_grades"] == [0.5, 0.2]


def test_select_asset_falls_back_to_nearest_grade_match(tmp_path: Path) -> None:
    near_audio = tmp_path / "go_near.wav"
    near_sidecar = tmp_path / "go_near.json"
    farther_audio = tmp_path / "go_farther.wav"
    farther_sidecar = tmp_path / "go_farther.json"
    near_audio.write_bytes(b"")
    farther_audio.write_bytes(b"")
    _write_sidecar(near_sidecar, ["G", "OW"])
    _write_sidecar(farther_sidecar, ["G", "OW"])

    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "word": "go",
                    "audio_path": near_audio.name,
                    "sidecar_path": near_sidecar.name,
                    "source_phonemes": ["G", "OW"],
                    "voice_mode": "woman",
                    "phoneme_values": [0.5, 0.3],
                },
                {
                    "word": "go",
                    "audio_path": farther_audio.name,
                    "sidecar_path": farther_sidecar.name,
                    "source_phonemes": ["G", "OW"],
                    "voice_mode": "woman",
                    "phoneme_values": [0.7, 0.7],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = select_asset(
        comparison=adapt_comparison_result(
            {
                "label_name": "go",
                "times": [0.25, 0.75],
                "per_time_l2": [0.5, 0.2],
            }
        ),
        candidates=load_asset_index(index_path),
        requested_voice="woman",
    )

    assert result["exact_match"] is False
    assert result["audio_path"] == str(near_audio.resolve())
    assert result["selected_phoneme_grades"] == [0.5, 0.3]


def test_select_asset_can_filter_by_generation_mode(tmp_path: Path) -> None:
    global_audio = tmp_path / "go_global.wav"
    global_sidecar = tmp_path / "go_global.json"
    per_audio = tmp_path / "go_per.wav"
    per_sidecar = tmp_path / "go_per.json"
    global_audio.write_bytes(b"")
    per_audio.write_bytes(b"")
    _write_sidecar(global_sidecar, ["G", "OW"])
    _write_sidecar(per_sidecar, ["G", "OW"])

    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "word": "go",
                    "audio_path": global_audio.name,
                    "sidecar_path": global_sidecar.name,
                    "source_phonemes": ["G", "OW"],
                    "voice_mode": "woman",
                    "generation_mode": "global",
                    "phoneme_values": [0.35],
                },
                {
                    "word": "go",
                    "audio_path": per_audio.name,
                    "sidecar_path": per_sidecar.name,
                    "source_phonemes": ["G", "OW"],
                    "voice_mode": "woman",
                    "generation_mode": "per_phoneme",
                    "phoneme_values": [0.5, 0.2],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = select_asset(
        comparison=adapt_comparison_result(
            {
                "label_name": "go",
                "times": [0.25, 0.75],
                "per_time_l2": [0.5, 0.2],
            }
        ),
        candidates=load_asset_index(index_path),
        requested_voice="woman",
        generation_mode=ASSET_MODE_PER_PHONEME,
    )

    assert result["generation_mode"] == ASSET_MODE_PER_PHONEME
    assert result["audio_path"] == str(per_audio.resolve())


def test_scan_asset_root_supports_mode_word_layout(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "per_phoneme" / "go" / "go_per_phoneme_0.5_0.2.json"
    audio_path = sidecar_path.with_suffix(".wav")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"")
    sidecar_path.write_text(
        json.dumps(
            {
                "word": "go",
                "transcript": "go",
                "output_audio_path": audio_path.name,
                "phoneme_values": [0.5, 0.2],
                "source_phonemes": ["G", "OW"],
                "metadata": {"voice_mode": "woman"},
                "source_alignment": [
                    {"phone": "G", "start_sec": 0.0, "end_sec": 0.5},
                    {"phone": "OW", "start_sec": 0.5, "end_sec": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = scan_asset_root(tmp_path)

    assert len(candidates) == 1
    assert candidates[0].generation_mode == ASSET_MODE_PER_PHONEME
    assert candidates[0].audio_path == audio_path.resolve()
