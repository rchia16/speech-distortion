# AGENTS.md — Coding-Agent Instructions for Distance-Aware Hybrid Speech Distortion

## Objective

Implement the hybrid speech distortion schema defined in `AGENTS.yaml` and explained in `PLANS.md`.

The system must support:

```yaml
jibberish:
  control_space: grapheme_guided
  validation_space: phoneme
  distance_space:
    - grapheme_distance
    - phoneme_distance
    - pronunciation_similarity
  render_space: acoustic

clarity:
  control_space: perceptual
  validation_space: phoneme_anchor_preservation
  distance_space:
    - acoustic_embedding_distance
    - spectral_feature_distance
  render_space: acoustic

timing_instability:
  control_space: perceptual
  validation_space: phone_order_and_syllable_shape
  distance_space:
    - temporal_distance
    - dtw_warp_distance
  render_space: temporal_audio
```

## Non-Negotiable Rules

1. Do not implement `jibberish` as random phoneme replacement.
2. Do not implement `jibberish` as pure spelling corruption.
3. Always validate grapheme variants through G2P or pronunciation lookup before rendering audio.
4. Keep the core pronunciation by default.
5. Full gibberish requires `allow_full_gibberish: true`.
6. `clarity` must not change phoneme identity.
7. `timing_instability` must not reorder phonemes.
8. Do not use a single global mismatch score for all sliders.
9. Use grapheme distance only for candidate ranking and UI/debugging.
10. Use phoneme distance and pronunciation similarity for safety.
11. Use acoustic embedding or spectral distance for clarity.
12. Use DTW warp and duration distance for timing instability.
13. Use neural embedding or Wasserstein distance only as supporting evidence, not as a direct command to corrupt pronunciation.
14. When distance signals disagree, preserve the word core.
15. When confidence is low, reduce distortion magnitude.

## Required Pipeline

```text
load target word
generate pronunciation candidates
extract pronunciation skeleton
parse slider envelopes
optionally DTW-align query to template
compute distance and embedding signals
route evidence to slider envelopes
generate grapheme candidates for jibberish
convert candidates to phones
score pronunciation similarity
select safe candidate
plan timing edits
repair unsafe edits
render or splice audio
apply clarity degradation
smooth and normalize
emit trace
```

## Distance and Embedding Responsibilities

```yaml
grapheme_distance:
  use_for:
    - candidate ranking
    - readability control
  do_not_use_for:
    - audio rendering authority

phoneme_distance:
  use_for:
    - pronunciation safety
    - candidate selection
    - word-core preservation

pronunciation_similarity:
  use_for:
    - thresholding
    - repair
    - short-word protection

acoustic_embedding_distance:
  use_for:
    - clarity estimation
    - audio similarity validation

temporal_distance:
  use_for:
    - timing instability estimation
    - pause/stutter/duration mismatch

neural_embedding_distance:
  use_for:
    - template mismatch evidence
    - confidence adjustment
  do_not_use_for:
    - direct full gibberish triggering
```

## Safety Defaults

```yaml
preserve_core_pronunciation: true
allow_full_gibberish: false
minimum_pronunciation_similarity: 0.72
minimum_pronunciation_similarity_short_word: 0.82
max_total_duration_drift_ratio: 0.12
```

## Acceptance Test Priorities

Implement tests for:

1. High jibberish still sounds like a strange version of the original word.
2. High clarity sounds slurred but does not change the word.
3. High timing instability creates stutters/stretching without changing phone order.
4. Dangerous grapheme edits are rejected or down-ranked.
5. High temporal distance routes to timing instability, not jibberish.
6. Low confidence reduces distortion magnitude.
7. Output traces expose grapheme, phoneme, distance, confidence, and edit decisions.
