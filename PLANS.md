# Pronunciation-Safe Plan: Jibberish, Clarity, and Timing Instability

## Goal

Keep `jibberish`, `clarity`, and `timing_instability` as separate sliders while preserving the core pronunciation identity of the target word.

The output may become distorted, slurred, temporally unstable, or mildly nonsensical, but it should still retain enough anchor structure that listeners can recognize the intended word, unless an explicit `allow_full_gibberish` override is enabled.

---

## Core Principle

Every word gets a pronunciation skeleton:

```text
word
  -> phoneme sequence
  -> anchor phones
  -> syllable nuclei
  -> stress pattern
  -> onset/rime shape
  -> allowed edit budget
```

The sliders may modify the surface realization, but the skeleton should remain mostly intact.

For example, for `no`:

```text
target phones: N OW
anchor phones: N, OW
protected structure: nasal onset + rounded vowel nucleus
```

A safe distorted version may sound like:

```text
no, noh, nuh-oh, n-aw, n-er-oh, n-oo
```

But should avoid becoming unrelated content like:

```text
tee-kah, bah-sha, siss, glorp
```

unless full gibberish mode is explicitly enabled.

---

## Pronunciation Preservation Layer

Add a new stage before slider execution:

```text
phoneme alignment
  -> pronunciation skeleton extraction
  -> protection map
  -> slider planners
  -> pronunciation-safe edit validator
  -> render
```

The protection map marks:

1. **Anchor consonants** — usually the first major consonant or onset.
2. **Vowel nucleus** — the main vowel or diphthong.
3. **Stress-bearing phones** — phones that carry the identity of the word.
4. **Word boundary shape** — whether the word starts/ends with consonant or vowel.
5. **Syllable count** — default should be preserved.
6. **Manner class** — nasal, stop, fricative, liquid, glide, vowel.

---

## Slider 1: Jibberish

### Updated meaning

`jibberish` controls pronunciation drift around the word, not total semantic replacement by default.

```text
0.0 = original word
0.3 = mild alternate pronunciation
0.6 = weird but still word-rooted
0.8 = heavily distorted but recognizable
1.0 = near-gibberish, with anchors still preserved by default
```

### Pronunciation-safe behavior

| Value | Behavior |
|---:|---|
| `0.00 - 0.25` | No phone replacement; only small vowel/color drift. |
| `0.25 - 0.50` | Nearby phone substitutions, preserving manner class. |
| `0.50 - 0.75` | Allow vowel variants and light inserted schwas, preserve anchor phones. |
| `0.75 - 1.00` | Strong mispronunciation, but keep onset/rime anchors and syllable nucleus. |

### Edit limits

Default limits:

```yaml
preserve_anchor_phones: true
preserve_primary_vowel_nucleus: true
preserve_syllable_count: true
max_phone_substitution_ratio: 0.40
max_insertions_per_syllable: 1
max_deletions_per_word: 0
max_anchor_phone_distance: 1
max_non_anchor_phone_distance: 3
```

For very short words like `yes`, `no`, `go`, and `stop`, use stricter limits:

```yaml
short_word_mode:
  max_phone_substitution_ratio: 0.25
  max_deletions_per_word: 0
  preserve_first_phone: true
  preserve_primary_vowel_nucleus: true
```

### Examples

For `no /N OW/`:

| Jibberish | Allowed examples |
|---:|---|
| `0.2` | `no`, `noh` |
| `0.4` | `nuh`, `naw` |
| `0.6` | `nuh-oh`, `ner-oh` |
| `0.8` | `nyoh`, `naw-uh` |
| `1.0` | `nuh-woh`, `nyer-oh`, but still starts near `N` and keeps a vowel nucleus |

Avoid:

```text
ka, sho, boo-tah, meh-see
```

because they lose the core pronunciation skeleton.

---

## Slider 2: Clarity

### Updated meaning

`clarity` changes articulatory precision while preserving the intended phoneme sequence.

```text
0.0 = crisp original word
1.0 = very slurred original word
```

### Pronunciation-safe behavior

Clarity may:

```text
soften consonant releases
reduce high-frequency detail
centralize vowels slightly
blur transitions
add nasal/breathy leakage
```

Clarity may not:

```text
replace anchor phones
delete the main vowel
change syllable count
move the word boundary shape beyond recognition
```

### Guardrails

```yaml
max_formant_centralization_for_anchor_vowels: 0.45
min_consonant_burst_energy_ratio: 0.35
min_vowel_duration_ratio: 0.70
max_transition_blur_ms: 60
```

For `no`, even high clarity should sound like a slurred `no`, not a different word.

---

## Slider 3: Timing Instability

### Updated meaning

`timing_instability` changes temporal delivery while preserving word order and phoneme identity.

```text
0.0 = stable timing
1.0 = unstable timing with repeats/pauses
```

### Pronunciation-safe behavior

Timing may:

```text
stretch phones
compress phones
insert brief hesitations
repeat onset fragments
create micro-stutters
```

Timing may not:

```text
reorder phones
delete anchor phones
repeat fragments so many times the word identity is lost
stretch a phone beyond recognizability
```

### Guardrails

```yaml
preserve_phone_order: true
max_phone_duration_scale: 1.8
min_phone_duration_scale: 0.55
max_onset_repeats: 2
max_pause_ms_inside_short_word: 180
max_total_duration_drift_ratio: 0.12
```

For `no`, acceptable outputs include:

```text
n-no
nnn-oh
no...oh
```

But avoid:

```text
n-n-n-n-n-o-o-o-o-o
```

unless an extreme experimental mode is enabled.

---

## Pronunciation Similarity Score

Add a validation score after planning and before rendering:

```text
pronunciation_similarity = weighted score from 0.0 to 1.0
```

Components:

```text
anchor_phone_preservation: 35%
vowel_nucleus_similarity: 25%
syllable_count_similarity: 15%
phone_order_similarity: 15%
duration_shape_similarity: 10%
```

Default acceptance threshold:

```yaml
minimum_pronunciation_similarity: 0.72
```

For short command words:

```yaml
minimum_pronunciation_similarity: 0.82
```

If a planned edit falls below threshold:

1. Restore anchor phones.
2. Reduce substitution distance.
3. Remove deletions.
4. Reduce insertions.
5. Reduce timing fragmentation.
6. Re-score.

---

## Revised Processing Pipeline

```text
1. Load word/template pronunciation.
2. Extract phoneme and syllable skeleton.
3. Mark protected anchor phones.
4. Parse jibberish, clarity, and timing envelopes.
5. Generate initial jibberish plan.
6. Generate timing plan.
7. Validate pronunciation similarity.
8. Repair unsafe edits.
9. Render or splice.
10. Apply clarity effects with anchor-preserving limits.
11. Smooth, normalize, and export.
```

---

## DTW Mismatch Handling

For mismatched signals like DTW plots:

```text
DTW residual does not directly mean full gibberish.
```

Instead:

```text
sustained phone-level mismatch
  -> increase jibberish locally, but keep anchors

spectral mismatch with stable timing
  -> increase clarity distortion locally

path slope or path irregularity
  -> increase timing instability locally
```

When mismatch is high, the system should still first produce a **word-rooted distortion**, not unrelated nonsense.

---

## Recommended Defaults

```yaml
jibberish:
  preserve_core_pronunciation: true
  allow_full_gibberish: false

clarity:
  preserve_core_pronunciation: true

timing_instability:
  preserve_core_pronunciation: true
```

---

## Override Mode

Only allow total gibberish if explicitly requested:

```yaml
allow_full_gibberish: true
minimum_pronunciation_similarity: 0.25
preserve_anchor_phones: false
preserve_syllable_count: false
```

This should not be the default.

---

## Acceptance Tests

### Test 1: High jibberish still preserves word core

Input:

```yaml
word: no
phones: [N, OW]
jibberish: 1.0
clarity: 0.0
timing_instability: 0.0
allow_full_gibberish: false
```

Expected:

```text
Output sounds like a strange variant of "no", not an unrelated word.
```

### Test 2: High clarity preserves phonemes

Input:

```yaml
word: stop
phones: [S, T, AA, P]
jibberish: 0.0
clarity: 1.0
timing_instability: 0.0
```

Expected:

```text
Output sounds like a slurred "stop"; S/T/AA/P structure remains.
```

### Test 3: High timing preserves order

Input:

```yaml
word: yes
phones: [Y, EH, S]
jibberish: 0.0
clarity: 0.0
timing_instability: 1.0
```

Expected:

```text
Output may stutter or stretch, but remains Y -> EH -> S.
```

### Test 4: Combined high values remain word-rooted

Input:

```yaml
word: pain
jibberish: 0.8
clarity: 0.7
timing_instability: 0.7
allow_full_gibberish: false
```

Expected:

```text
Output is distorted, slurred, and unstable, but still resembles "pain".
```
