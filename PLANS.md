# PLANS.md — Hybrid Grapheme-Guided, Phoneme-Safe Speech Distortion with Distance/Embedding Layer

## Purpose

This plan defines a hybrid speech distortion system that is intuitive to control while preserving the core pronunciation of each target word.

The system uses:

```text
grapheme-guided controls
  -> distance and embedding evidence
  -> phoneme validation
  -> acoustic and timing rendering
```

The three primary sliders remain separate:

1. `jibberish`
2. `clarity`
3. `timing_instability`

Each slider consumes different evidence:

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
    - formant_trajectory_distance
  render_space: acoustic

timing_instability:
  control_space: perceptual
  validation_space: phone_order_and_syllable_shape
  distance_space:
    - temporal_distance
    - dtw_warp_distance
    - duration_shape_distance
  render_space: temporal_audio
```

---

## Core Design Rule

Do not use one global mismatch score for all distortions.

Instead:

```text
distance tells the system what kind of mismatch occurred;
the slider decides how that mismatch becomes speech distortion;
pronunciation validation prevents the output from losing the word core.
```

This avoids bad mappings such as:

```text
high DTW residual -> high jibberish
```

when the true issue may only be timing instability.

---

## Why Hybrid Instead of Pure Grapheme or Pure Phoneme

A pure phoneme system is accurate but less intuitive for users.

A pure grapheme system is intuitive but unsafe for audio because spelling changes do not reliably map to pronunciation changes.

Examples:

```text
no -> noh       safe, close pronunciation
no -> noo       close but vowel quality changes
no -> now       small spelling edit, larger pronunciation shift
bath -> bathe   spelling edit changes vowel and final consonant
```

Therefore:

```text
Graphemes = user control, candidate generation, UI traces.
Phonemes = validation, anchors, pronunciation safety.
Acoustics = clarity, slurring, muffling.
Temporal audio = timing instability, stutters, duration changes.
Distances/embeddings = evidence routing and scoring.
```

---

## Processing Pipeline

```text
1. Input target word.
2. Generate canonical grapheme representation.
3. Generate pronunciation candidates using dictionary/G2P.
4. Extract pronunciation skeleton.
5. Parse jibberish, clarity, and timing_instability values or envelopes.
6. Optionally DTW-align query signal to template time.
7. Compute distance and embedding signals.
8. Route distance evidence into separate slider envelopes.
9. Generate grapheme-guided jibberish candidates.
10. Convert grapheme candidates to phonemes.
11. Score candidates against pronunciation skeleton.
12. Select a safe phone-level edit plan.
13. Apply timing instability over phones/syllables.
14. Render or splice audio.
15. Apply clarity effects acoustically.
16. Smooth, normalize, and export.
17. Emit a trace showing grapheme, phoneme, distance, confidence, and edit decisions.
```

---

## Distance and Embedding Layer

The system should compute multiple distance signals and route them to the appropriate slider.

```text
target word
  -> grapheme candidates
  -> G2P candidates
  -> grapheme distance
  -> phoneme distance
  -> pronunciation similarity
  -> safe candidate selection

query/template signals
  -> DTW alignment
  -> acoustic distance
  -> temporal distance
  -> optional neural distance
  -> slider envelope estimation

selected candidate + slider envelopes
  -> timing plan
  -> render/splice
  -> clarity layer
  -> output audio + trace
```

### Distance Types

```yaml
grapheme_distance:
  purpose:
    - rank spelling-like variants
    - discourage unreadable spelling drift
  examples:
    - weighted Levenshtein distance
    - repeated-letter penalty
    - onset-preservation penalty
    - rime-preservation penalty

phoneme_distance:
  purpose:
    - validate pronunciation safety
    - preserve word core
    - rank candidate pronunciations
  examples:
    - articulatory feature distance
    - phoneme edit distance
    - vowel nucleus distance
    - manner/place/voicing distance

pronunciation_similarity:
  purpose:
    - reject unsafe candidates
    - repair excessive edits
    - enforce short-word safety
  range: [0.0, 1.0]

acoustic_embedding_distance:
  purpose:
    - estimate clarity degradation
    - compare audio similarity
    - separate acoustic mismatch from timing mismatch
  examples:
    - wav2vec/HuBERT cosine distance
    - MFCC DTW distance
    - formant trajectory distance
    - spectral feature distance

temporal_distance:
  purpose:
    - estimate timing instability
    - detect stutters, holds, pauses, and duration mismatch
  examples:
    - DTW warp distance
    - local slope deviation
    - duration-shape distance
    - pause-pattern distance

neural_embedding_distance:
  purpose:
    - interpret EEG/template mismatch
    - support confidence or prior estimation
    - aid cross-subject or cross-session calibration
  examples:
    - Wasserstein distribution distance
    - cosine distance between EEG embeddings
    - Mahalanobis distance between template features
```

### Routing

```yaml
linguistic_mismatch:
  evidence:
    - high_phoneme_distance
    - low_pronunciation_similarity
    - sustained_phone_level_residual
  output_slider: jibberish

acoustic_mismatch:
  evidence:
    - high_acoustic_embedding_distance
    - high_spectral_feature_distance
    - stable_phone_identity
  output_slider: clarity

temporal_mismatch:
  evidence:
    - high_temporal_distance
    - abnormal_dtw_warp_ratio
    - pause_or_hold_mismatch
  output_slider: timing_instability

neural_template_mismatch:
  evidence:
    - high_neural_embedding_distance
    - high_wasserstein_distribution_distance
  output:
    - local_confidence_adjustment
    - optional_slider_prior
```

Important safety rule:

```yaml
high_neural_or_dtw_distance_alone_must_not_trigger_full_gibberish: true
```

---

## Confidence and Evidence

Every inferred slider value should have:

```yaml
value: 0.0-1.0
confidence: 0.0-1.0
evidence_sources:
  - distance_or_embedding_name
```

Example:

```yaml
inferred_controls:
  jibberish:
    value: 0.62
    confidence: 0.71
    evidence_sources:
      - phoneme_distance
      - pronunciation_similarity

  clarity:
    value: 0.35
    confidence: 0.64
    evidence_sources:
      - acoustic_embedding_distance
      - spectral_feature_distance

  timing_instability:
    value: 0.78
    confidence: 0.83
    evidence_sources:
      - dtw_warp_distance
      - pause_pattern_distance
```

When confidence is low:

```text
reduce slider magnitude
preserve pronunciation more strictly
avoid full gibberish
prefer conservative candidate selection
```

---

## Pronunciation Skeleton

Every target word must produce a protected pronunciation skeleton.

Example for `no`:

```yaml
target_word: "no"
canonical_graphemes: "no"
canonical_phones: ["N", "OW"]
anchor_phones: ["N", "OW"]
primary_vowel_nucleus: "OW"
syllable_count: 1
onset_rime_shape: "CV"
word_boundary_shape:
  starts_with: consonant
  ends_with: vowel
protected_features:
  - nasal_onset
  - rounded_vowel_nucleus
  - single_syllable_shape
```

The skeleton prevents unrelated output such as:

```text
ka
sho
boo-tah
meh-see
```

unless `allow_full_gibberish: true`.

---

## Slider: Jibberish

### Meaning

`jibberish` controls spelling-like and pronunciation-like drift around the original word.

```text
0.00 = original word
0.25 = mild spelling/pronunciation drift
0.50 = word-rooted mispronunciation
0.75 = strong but recognizable word-rooted gibberish
1.00 = near-gibberish while retaining anchors by default
```

### Distance Inputs

```yaml
distance_inputs:
  primary:
    - phoneme_distance
    - pronunciation_similarity
  secondary:
    - grapheme_distance
  optional:
    - word_or_syllable_embedding_distance
  exclude:
    - pure_acoustic_noise_distance
    - pure_timing_distance
```

### Implementation

```text
target grapheme
  -> grapheme variant generator
  -> grapheme distance ranking
  -> G2P / pronunciation lookup
  -> phoneme distance scoring
  -> pronunciation similarity validation
  -> safe candidate selection
  -> render plan
```

### Candidate Generation

Allowed operations:

```text
letter repetition
vowel spelling variants
soft consonant insertions
weak schwa-like endings
onset-preserving mutations
rime-preserving mutations
pseudo-syllable expansion
```

For `no`, safe-ish candidates include:

```text
no, noh, noo, nuh, naw, nyoh, nwoh, nuh-oh, nyuh-oh
```

### Candidate Selection

```yaml
candidate_selection:
  rank_by:
    - pronunciation_similarity
    - phoneme_distance
    - grapheme_distance
  reject_if:
    pronunciation_similarity_below_threshold: true
  prefer:
    - onset_preserving_candidates
    - vowel_nucleus_preserving_candidates
    - syllable_count_preserving_candidates
```

### Guardrails

```yaml
preserve_anchor_phones: true
preserve_primary_vowel_nucleus: true
preserve_syllable_count: true
preserve_phone_order: true
max_phone_substitution_ratio: 0.40
max_insertions_per_syllable: 1
max_deletions_per_word: 0
max_anchor_phone_distance: 1
max_non_anchor_phone_distance: 3
```

Short-word mode:

```yaml
enabled_when_phone_count_lte: 3
max_phone_substitution_ratio: 0.25
max_deletions_per_word: 0
preserve_first_phone: true
preserve_primary_vowel_nucleus: true
minimum_pronunciation_similarity: 0.82
```

---

## Slider: Clarity

### Meaning

`clarity` controls acoustic/articulatory precision. It must not change phoneme identity.

```text
0.00 = crisp pronunciation
0.25 = slightly softened articulation
0.50 = noticeably blurred/slurred
0.75 = strongly muffled or imprecise
1.00 = very mushy but still word-rooted
```

### Distance Inputs

```yaml
distance_inputs:
  primary:
    - acoustic_embedding_distance
    - spectral_feature_distance
  secondary:
    - formant_trajectory_distance
    - fricative_band_energy_distance
    - consonant_burst_energy_distance
  exclude:
    - grapheme_distance
    - phoneme_identity_distance
```

### Estimation Rule

```yaml
clarity_should_increase_when:
  - acoustic_distance_high
  - phoneme_identity_stable
  - timing_distance_low_or_moderate
```

### Effects

```text
formant centralization
spectral tilt
high-frequency reduction
consonant burst softening
fricative smearing
transition blur
nasal leakage
breath/noise mix
```

### Guardrails

```yaml
preserve_anchor_phones: true
preserve_primary_vowel_nucleus: true
preserve_syllable_count: true
max_formant_centralization_for_anchor_vowels: 0.45
min_consonant_burst_energy_ratio: 0.35
min_vowel_duration_ratio: 0.70
max_transition_blur_ms: 60
preserve_anchor_phone_audibility: true
```

---

## Slider: Timing Instability

### Meaning

`timing_instability` controls temporal irregularity while preserving phone order.

```text
0.00 = stable timing
0.25 = mild phone duration jitter
0.50 = uneven rhythm
0.75 = stutters and pauses
1.00 = fragmented but ordered word delivery
```

### Distance Inputs

```yaml
distance_inputs:
  primary:
    - temporal_distance
    - dtw_warp_distance
  secondary:
    - duration_shape_distance
    - pause_pattern_distance
  exclude:
    - grapheme_distance
    - phoneme_substitution_distance
```

### Estimation Rule

```yaml
timing_instability_should_increase_when:
  - local_dtw_slope_deviates
  - phone_duration_ratio_extreme
  - pause_or_hold_detected
  - repeated_fragment_detected
```

### Effects

```text
phone duration jitter
syllable duration jitter
pause insertion
onset repetition
micro-stutters
fragment repeats
tempo instability
```

### Guardrails

```yaml
preserve_phone_order: true
preserve_anchor_phones: true
max_phone_duration_scale: 1.8
min_phone_duration_scale: 0.55
max_onset_repeats: 2
max_pause_ms_inside_short_word: 180
max_total_duration_drift_ratio: 0.12
max_fragment_repetition_count: 2
```

---

## Pronunciation Similarity

Before rendering, validate each plan.

```yaml
default_minimum: 0.72
short_word_minimum: 0.82
weights:
  anchor_phone_preservation: 0.35
  vowel_nucleus_similarity: 0.25
  syllable_count_similarity: 0.15
  phone_order_similarity: 0.15
  duration_shape_similarity: 0.10
```

Repair order if the score fails:

```text
restore anchor phones
reduce phoneme substitution distance
remove deletions
limit insertions
reduce repetitions
reduce timing fragmentation
rescore
```

---

## Output Trace

Emit this whenever possible:

```yaml
trace:
  target_word: "no"
  target_graphemes: "no"
  target_phones: ["N", "OW"]
  selected_grapheme_variant: "nuh-oh"
  selected_phones: ["N", "AH", "OW"]
  pronunciation_similarity: 0.84
  slider_values:
    jibberish: 0.75
    clarity: 0.35
    timing_instability: 0.40
  confidence:
    jibberish: 0.71
    clarity: 0.64
    timing_instability: 0.83
  distance_evidence:
    grapheme_distance: 0.42
    phoneme_distance: 0.36
    acoustic_embedding_distance: 0.35
    temporal_distance: 0.78
  edits:
    - type: vowel_shift
      from: "OW"
      to: "AH+OW"
    - type: weak_syllable_expansion
      inserted: "AH"
    - type: timing_jitter
      scope: vowel_nucleus
```

---

## Acceptance Tests

### 1. Grapheme-guided jibberish preserves word core

```yaml
word: "no"
jibberish: 1.0
clarity: 0.0
timing_instability: 0.0
allow_full_gibberish: false
```

Expected: a strange version of `no`, not an unrelated word.

### 2. Clarity does not alter phoneme sequence

```yaml
word: "stop"
jibberish: 0.0
clarity: 1.0
timing_instability: 0.0
```

Expected: slurred `stop`; phone order remains `S T AA P`.

### 3. Timing preserves phone order

```yaml
word: "yes"
jibberish: 0.0
clarity: 0.0
timing_instability: 1.0
```

Expected: stuttered or stretched `yes`; phone order remains `Y EH S`.

### 4. Dangerous grapheme edit is rejected or down-ranked

```yaml
word: "bath"
candidate_grapheme: "bathe"
```

Expected: reject or down-rank if pronunciation similarity is too low.

### 5. Distance routing is independent

```yaml
evidence:
  high_temporal_distance: true
  low_phoneme_distance: true
  low_acoustic_distance: true
```

Expected: increase `timing_instability`, not `jibberish` or `clarity`.

---

## Override Mode

Full gibberish must be explicit.

```yaml
allow_full_gibberish: true
minimum_pronunciation_similarity: 0.25
preserve_anchor_phones: false
preserve_syllable_count: false
```
