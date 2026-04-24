**Architecture**
`audio + transcript -> phoneme alignment -> error planner -> dual-path segment editor -> timbre projection -> overlap-add stitcher -> transformed audio`

**1. Alignment Layer**
Use a fast CTC forced aligner as the runtime path, with an offline high-accuracy aligner only for calibration.

- Runtime aligner: `torchaudio` forced alignment with `Wav2Vec2FABundle` / `forced_align()`. This is the best fit for near-real-time because it is local, open, and designed for transcript-to-audio alignment. Source: [torchaudio forced alignment](https://docs.pytorch.org/audio/2.8.0/tutorials/forced_alignment_tutorial.html)
- Offline calibration/evaluation aligner: `Montreal Forced Aligner` for checking boundary quality and tuning the fast aligner’s postprocessing rules. MFA is accurate, but it is not the latency path. Source: [MFA GitHub](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)

Output of this layer:
- word boundaries
- phone boundaries
- syllable spans
- stress markers
- confidence per aligned phone

**2. Linguistic Representation Layer**
Convert the transcript into a phone graph with feature metadata.

- G2P/phonemization: `phonemizer` backed by `espeak-ng`, or direct `espeak-ng` phoneme output. Sources: [phonemizer docs](https://bootphon.github.io/phonemizer/), [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
- Attach to each phone:
  - place, manner, voicing
  - vowel height/backness/rounding
  - sonority
  - later-developing-phone flag
  - cluster membership
  - syllable position
  - word-level complexity score

This gives you a phone stream that can drive both categorical edits and “distorted” realizations.

**3. Complexity-Aware Error Planner**
This is the core controller. It decides what to do to each phone, conditioned on severity and articulatory complexity.

Inputs:
- aligned phone sequence
- word complexity score
- user severity profile
- local prosodic structure

Outputs:
- edit plan per phone span:
  - `keep`
  - `substitute`
  - `distorted_substitute`
  - `add`
  - `distort`
  - `lengthen_vowel`
  - `lengthen_consonant`
  - `compensatory_shorten`

Complexity score should combine:
- consonant cluster count/size
- later-developing phoneme presence
- phonotactic complexity
- syllable count
- stress pattern complexity

Rule examples:
- simple substitution: `/r/ -> /w/`, `/k/ -> /t/`
- addition: insert schwa before difficult onset cluster, `blue -> buhlue`
- distortion: lateralized `/s/`, nasalized stop, reduced stop closure precision
- distorted substitution: `/s/ -> off-target slushy alveopalatal fricative`
- increased distortion: raise probability and magnitude of edits when complexity score is high, especially in clusters like `str`, `spl`, `skr`

**4. Dual-Path Segment Editor**
You need two editing paths because one method will not cover all requested errors while preserving speaker identity.

**Path A: In-place source-domain editing**
Use this for:
- sound distortions
- vowel lengthening
- consonant lengthening
- mild off-target segment shaping

Recommended tools:
- `Parselmouth` / Praat for source-filter style manipulations and duration changes. Source: [Parselmouth](https://parselmouth.readthedocs.io/en/latest/api/parselmouth.html)
- `psola` for local TD-PSOLA time stretching / pitch-safe duration edits. Source: [psola](https://pypi.org/project/psola/)
- optionally `pyworld` / WORLD-style decomposition if you want explicit control over spectral envelope and F0 during edits

Use Path A for:
- vowel lengthening independent of speaking rate
- consonant frication lengthening
- local formant centralization
- spectral tilt changes
- nasal leakage / antiresonance approximations
- slushy /s/-like distortions by local spectral reshaping toward alveopalatal energy concentration

Why:
- It best preserves the original speaker and local coarticulation
- It is the lowest-latency path

**Path B: Segment resynthesis**
Use this for:
- substitutions
- distorted substitutions
- additions
- severe complexity-driven failures where source editing is insufficient

Recommended base synthesizer:
- `XTTS` as the open local speech generator, because it supports voice cloning and streaming inference with documented low latency. Source: [XTTS docs](https://docs.coqui.ai/en/latest/models/xtts.html)

Path B flow:
- resynthesize only short edited spans, not the whole utterance
- synthesize phones or micro-phrases containing the target changed segment plus one-phone context on each side
- generate from edited phoneme/text representation, not from raw graphemes alone

Examples:
- `rabbit -> wabbit`: resynthesize `/ræb/` or `/r/`-anchored micro-span with `/w/`
- `blue -> buhlue`: synthesize inserted schwa-containing onset span
- `str -> slurry distorted cluster`: synthesize a context span with cluster simplification plus degraded fricative realization

**5. Timbre Projection Layer**
Pure TTS fragments will drift from the source speaker. To preserve timbre, add a local voice-conversion projection step after Path B.

Recommended tool:
- `RVC` for projecting synthesized fragments back toward the original speaker’s timbre. Sources: [RVC library repo](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion), [RVC WebUI repo](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

Use it only on Path B fragments:
- source for VC = synthesized edited fragment
- target voice = model trained on the original speaker or a small speaker cache
- result = edited content with voice closer to the input speaker

Why this matters:
- XTTS gives you phonological freedom
- RVC pulls the fragment back toward the source timbre
- the combination is much better than either alone for your use case

**6. Duration Budget Manager**
This handles your prosodic requirement: local segment lengthening without changing overall utterance duration.

For each prosodic edit:
- define a local budget window, usually the current syllable plus adjacent unstressed phones
- lengthen target phone by `+d`
- shorten neighboring phones by `-d` total
- preserve word and utterance duration within tolerance

Rules:
- vowel lengthening:
  - take time mainly from adjacent unstressed vowels, interconsonantal schwa-like regions, or following sonorants
- consonant lengthening:
  - for fricatives and nasals, stretch internal steady-state region
  - for stops, extend closure, not release burst
- do not globally slow the utterance
- cap compensatory shortening to avoid impossible phones

This module should run before final stitching so both source-edited and synthesized spans honor the same timing grid.

**7. Stitching and Crossfade Layer**
Reassemble the utterance from unedited spans plus edited spans.

Use:
- phone-aware cut points near low-energy boundaries
- short crossfades
- optional phase-aware overlap-add for voiced regions
- energy and F0 smoothing across joins

This layer is critical because most failures in a prototype like this are audible seams, not bad phonology.

**How the requested distortion classes map into the stack**

- Simple substitutions:
  - planner marks phone as `substitute`
  - Path B resynthesizes local span
  - timbre projection with RVC
- Distorted substitutions:
  - planner marks `distorted_substitute`
  - generate substitute phone with neighboring intended-phone features retained
  - then apply Path A distortion shaping to the resynthesized fragment
- Additions:
  - planner inserts epenthetic phone, usually schwa or consonant release
  - Path B resynthesizes the local cluster span
  - duration budget manager compresses nearby phones
- Distortions:
  - Path A only
  - spectral / formant / nasality / frication-shape edits on the original segment
- Increased distortions with articulatory complexity:
  - planner scales both probability and severity using complexity score
  - hard words get more cluster reduction, more off-target shaping, or more severe distortions
- Lengthened vowels independent of rate:
  - Path A local stretch plus compensatory shortening
- Lengthened consonants independent of rate:
  - Path A local stretch of closure or frication plus compensatory shortening

**Why this stack fits your constraints**
- fully local and open weights
- preserves speaker better than pure TTS
- supports both phonological errors and motor-like distortions
- can reach under 1 second for short utterances if:
  - alignment uses GPU CTC forced alignment
  - only short spans are resynthesized
  - most edits stay on Path A
  - RVC is applied only to edited fragments, not the full file

**Important constraint**
A strict `< 1s` budget is realistic for short utterances and sparse edits, not for arbitrary long recordings with many substitutions. The architecture still fits; the latency budget just depends on edit density.

**Recommended stack summary**
- Alignment: `torchaudio forced_align` runtime, `MFA` offline validation
- G2P/phones: `phonemizer` + `espeak-ng`
- In-place edits: `Parselmouth` + `psola`
- Fragment synthesis: `Coqui XTTS`
- Timbre recovery: `RVC`
- Orchestration: phone-graph planner + duration budget manager + stitcher

Sources:
- [Torchaudio forced alignment](https://docs.pytorch.org/audio/2.8.0/tutorials/forced_alignment_tutorial.html)
- [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
- [Phonemizer](https://bootphon.github.io/phonemizer/)
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
- [Parselmouth](https://parselmouth.readthedocs.io/en/latest/api/parselmouth.html)
- [psola](https://pypi.org/project/psola/)
- [Coqui XTTS](https://docs.coqui.ai/en/latest/models/xtts.html)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)
- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

If you want, the next step can be a module-by-module repo design for this architecture without writing implementation yet.
