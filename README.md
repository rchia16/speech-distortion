# Speech Distortion GUI

Desktop GUI and supporting pipeline code for speech augmentation experiments with:

- static noise
- volume dropout
- phone style noise mixing
- blabber phoneme mutation with Coqui TTS

The GUI supports both global quality control and per-phoneme quality control. For signal-domain modes, per-phoneme mode applies augmentation to inferred phoneme segments. For `blabber`, both global and per-phoneme modes now resolve substitutions from a shared phoneme-distance reference.

## Repo Layout

- `speech_distortion_gui.py`: main desktop GUI
- `build_blabber_library.py`: standalone batch generator for blabber assets
- `select_blabber_asset.py`: local selector that maps template mismatch outputs onto pre-generated blabber assets
- `export_phoneme_distances.py`: exports the shared phoneme-distance reference
- `src/speech_distortion_pipeline/phonology/blabber_config.py`: shared Blabber constants used by the GUI and library builder
- `src/speech_distortion_pipeline/phonology/phone_distance_reference.py`: shared deterministic distance resolvers for global and per-phoneme Blabber
- `src/speech_distortion_pipeline/phonology/acn_neighbors.py`: ACN embedding-based phone and phone-sequence distance ranking
- `tests/`: targeted tests for phoneme-distance loading and ranking helpers
- `environment.coqui-blabber.yaml`: recommended conda environment for GUI and Coqui support

## Requirements

- Windows with Python/Conda available
- A working Conda installation
- `coqui-blabber` environment created from the provided YAML

## Setup

```powershell
conda env create -f environment.coqui-blabber.yaml
conda activate coqui-blabber
```

If you prefer to run the GUI from another environment while using Coqui in `coqui-blabber`, set:

```powershell
$env:BLABBER_CONDA_ENV="coqui-blabber"
```

Shared Blabber runtime constants live in `src/speech_distortion_pipeline/phonology/blabber_config.py`.
Shared ladder counts live in `src/speech_distortion_pipeline/phonology/phone_distance_reference.py`:

- `GLOBAL_BLABBER_PRESET_COUNT`
- `PER_PHONEME_BLABBER_PRESET_COUNT`

The following environment variables override the default Coqui settings:

```powershell
$env:BLABBER_COQUI_WOMAN_MODEL="tts_models/en/ljspeech/tacotron2-DDC_ph"
$env:BLABBER_COQUI_MAN_MODEL="tts_models/en/vctk/vits"
$env:BLABBER_COQUI_CLONE_MODEL="tts_models/multilingual/multi-dataset/xtts_v2"
```

`BLABBER_COQUI_MODEL` is still accepted as a backward-compatible alias for the woman preset when
`BLABBER_COQUI_WOMAN_MODEL` is not set.

## Model Downloads

There are two different model sources used by this repo:

- the Apple ACN embedder used by `src/speech_distortion_pipeline/phonology/acn_neighbors.py`
- the Coqui TTS and voice-conversion models referenced by the `BLABBER_COQUI_*` environment variables above

### ACN Embedder Model

`acn_neighbors.py` expects the Apple Acoustic Neighbor Embeddings text embedder model to be available locally.

Download source:

- Apple research page: <https://machinelearning.apple.com/research/acoustic-neighbor-embeddings>
- Apple model/package repo: <https://github.com/apple/ml-acn-embed>

What to download:

- the pretrained ACN archive `model.tgz`

Where to put it:

- option 1: place `model.tgz` in the repo root, so the code can auto-extract it on first use
- option 2: extract it yourself so the model directory exists at `model/embedder-64/`
- option 3: store it somewhere else and set `ACN_EMBED_MODEL_DIR` to that extracted `embedder-64` directory

Expected local layouts:

```text
speech-distortion/
  model.tgz
```

or

```text
speech-distortion/
  model/
    embedder-64/
```

Example override:

```powershell
$env:ACN_EMBED_MODEL_DIR="D:\models\acn\embedder-64"
```

### Coqui TTS Models

The Coqui model values in `BLABBER_COQUI_WOMAN_MODEL`, `BLABBER_COQUI_MAN_MODEL`, and
`BLABBER_COQUI_CLONE_MODEL` are model IDs, not repo-local files. With `coqui-tts` installed, Coqui
downloads those models into its own cache the first time they are used.

Default model IDs used by this repo:

- `tts_models/en/ljspeech/tacotron2-DDC_ph` for the `woman` preset
- `tts_models/en/vctk/vits` for the `man` preset
- `tts_models/multilingual/multi-dataset/xtts_v2` for `source_clone`

For woman-to-male post-render style transfer, the repo now also supports the Coqui voice-conversion model:

- `voice_conversion_models/multilingual/multi-dataset/openvoice_v2`

Experimental direct phoneme render backends are controlled separately:

```powershell
$env:BLABBER_PHONEME_VITS_MODEL="your phoneme-trained Coqui VITS checkpoint"
$env:BLABBER_FASTPITCH_MODEL="nvidia/tts_en_fastpitch"
$env:BLABBER_FASTPITCH_HIFIGAN_MODEL="nvidia/tts_hifigan"
$env:BLABBER_KOKORO_MODEL="hexgrad/Kokoro-82M"
$env:BLABBER_KOKORO_VOICE="bf_emma"
```

Notes:

- `BLABBER_PHONEME_VITS_MODEL` must point to a phoneme-trained Coqui VITS model, not a generic text-first VITS checkpoint
- `fastpitch` now uses NeMo directly when `nemo_toolkit` is installed
- the default FastPitch pair is `nvidia/tts_en_fastpitch` plus `nvidia/tts_hifigan`
- FastPitch currently renders from phoneme-derived surrogate text, not raw IPA tokens
- `BLABBER_FASTPITCH_RENDERER` is still accepted as an escape hatch for a custom external adapter if needed
- `kokoro` uses the `kokoro` Python package with `KPipeline`, not Coqui
- the current Kokoro GUI voices are `bf_emma` and `gm_fable`; `gm_fable` is a GUI alias that maps to Kokoro's published `bm_fable` voice ID
- Kokoro follows the published pronunciation-markup path like `[text](/ipa/)`, so install `kokoro`, `soundfile`, and the English fallback dependency `espeak-ng` in the runtime that launches the GUI or builder

## Style Transfer Presets

Woman-to-male style transfer is configured through:

- `config/style_transfer_presets.json`

This file defines named target voice presets for post-render conversion. The GUI and
`build_blabber_library.py` both read the same manifest.

Expected shape:

```json
{
  "presets": [
    {
      "name": "male_demo",
      "display_name": "Male Demo",
      "target_gender": "male",
      "target_wavs": [
        "reference-voices/male_demo_01.wav",
        "reference-voices/male_demo_02.wav"
      ],
      "description": "Optional note shown only in the config."
    }
  ]
}
```

Notes:

- `target_wavs` can contain one or more local male reference WAVs
- relative paths are resolved relative to `config/style_transfer_presets.json`
- the checked-in file currently starts empty, so add your own presets before using the backend
- the current post-render backend supports only woman-to-male conversion presets

## Run The GUI

```powershell
python speech_distortion_gui.py
```

The default demo input is `yes_slow.wav` when present in the repo root.

## GUI Modes

### Static Noise

- `0.0` = only noise
- `1.0` = original signal

### Volume Dropout

- lower quality increases dropout coverage
- per-phoneme mode applies dropout independently across inferred phoneme segments

### Phone Style

- mixes speech with white or pink noise
- supports global quality or per-phoneme quality

### Blabber

- requires a single-word transcript
- uses heuristic G2P to derive the source phoneme sequence
- supports preset `woman` and `man` TTS outputs plus optional source voice cloning through Coqui `speaker_wav`
- supports optional post-render woman-to-male style transfer through named presets in `config/style_transfer_presets.json`
- supports experimental woman-path phoneme backends `coqui_tacotron2_ddc_ph`, `fastpitch`, and `kokoro`
- when `Phoneme TTS = kokoro`, the GUI enables a backend-specific `Kokoro voice` selector for `bf_emma` or `gm_fable`
- provides a `Generate Sequence` button to preview the resolved phone sequence before rendering
- caches the generated sequence and reuses it for the next render when inputs have not changed

#### Global Blabber

- global mode uses a deterministic per-word distance ladder built from `phoneme_distances.json`
- each quality value maps to a preset index, and each preset index resolves to the next distinct word-level substitution state by cumulative distance
- the global max phoneme distance control sets the quality-0 word-distance endpoint; global quality gradations are rescaled from total distance 0 to that max, while individual candidates above the max are excluded
- rendered metadata includes:
  - `global_preset_index`
  - `global_distance_total`
  - `global_max_phoneme_distance`
  - `expansion_used`

#### Per-Phoneme Blabber

- each slider controls only its matching source phoneme
- each phoneme resolves through its own deterministic distance ladder using the same shared reference file
- per-phoneme ladders now prioritize single-phone substitutions first and only introduce multi-phone expansions at the harshest presets
- per-phoneme renders now include:
  - `per_phone_preset_indices`
  - `per_phone_distances`
  - `per_phone_distance_total`
  - `whole_word_distance_total`
  - `per_phone_sequences`
  - `per_phone_numeric_entries_used`
  - `per_phone_reference_warning`

#### Distance Reference Behavior

- the preferred source of truth is `phoneme_distances.json` with `ranked_candidate_entries_near_to_far`
- each candidate entry includes:
  - replacement phone sequence
  - numeric distance
  - rank
- if numeric entries are missing, the runtime falls back to rank-based synthetic distances derived from candidate order and marks per-phoneme renders with a reference warning

## Standalone Blabber Library Builder

`build_blabber_library.py` generates standalone Blabber assets from a manifest of input WAVs and single-word transcripts.
The builder renders the female/woman preset voice first and can now optionally apply a post-process woman-to-male
style-transfer stage on the generated audio.
For `global_soft` generation, `--global-max-phoneme-distance` applies the same per-phone distance cap as the GUI.

Basic run flow:

1. Create a manifest with `audio_path`, `transcript`, and `phoneme_count`.
2. Run the builder against that manifest.
3. Read outputs from the generated `global/` and `per_phoneme/` subdirectories plus the root summary JSON files.

Supported manifest formats:

- `.csv`
- `.tsv`
- `.json`
- `.jsonl`

Required fields per row:

- `audio_path`
- `transcript`
- `phoneme_count`

Example CSV:

```csv
audio_path,transcript,phoneme_count
yes_slow.wav,yes,3
```

Default run:

```powershell
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets
```

This default command:

- generates both `global` and `per_phoneme` banks
- uses `kokoro` as the phoneme backend
- uses Kokoro voice `bf_emma`
- writes an asset index and per-mode summaries at the output root

Common variants:

```powershell
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets --generation-mode global_soft
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets --generation-mode per_phoneme
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets --kokoro-voice gm_fable
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets --style-transfer-backend openvoice_v2 --style-transfer-target-voice male_demo
```

Expected output layout:

```text
speech-assets/
  blabber_asset_index.json
  global_summary.json
  per_phoneme_summary.json
  global/
    yes/
      yes_global_p00_q1.wav
      yes_global_p00_q1.json
      ...
  per_phoneme/
    yes/
      yes_per_phoneme_1_0_0.wav
      yes_per_phoneme_1_0_0.json
      ...
```

Current behavior:

- default generation mode is `both`, which writes separate `global` and `per_phoneme` banks in one run
- default direct phoneme render backend is `kokoro`
- default builder Kokoro voice is `bf_emma`
- global generation produces `GLOBAL_BLABBER_PRESET_COUNT` ordered outputs per word
- per-phoneme deterministic ladders use `PER_PHONEME_BLABBER_PRESET_COUNT`
- per-phoneme generation remains available as the original combinatorial mode
- generated Blabber metadata includes global or per-phone distance information depending on generation mode
- generated Blabber metadata now also includes `voice_mode`, `coqui_model_name`, and style-transfer status fields
- generated assets are organized as:
  - `speech-assets/global/<word>/...`
  - `speech-assets/per_phoneme/<word>/...`
- the output root now also writes:
  - `blabber_asset_index.json`
  - `global_summary.json`
  - `per_phoneme_summary.json`
- each generated asset also writes a `.json` sidecar with:
  - `phoneme_values`
  - `source_phonemes`
  - `source_alignment`
  - `output_alignment`
  - `output_audio_path`

Notes:

- `transcript` must currently be a single word
- `phoneme_count` must exactly match the detected source phoneme count for that word
- Coqui must be installed in the configured `BLABBER_CONDA_ENV`
- the builder currently uses only the configured woman preset model
- `--generation-mode both|global_soft|per_phoneme` controls whether one or both banks are written
- `--phoneme-render-backend` can be used to try `coqui_tacotron2_ddc_ph`, `fastpitch`, `kokoro`, or `phoneme_vits`
- `--kokoro-voice bf_emma|gm_fable` overrides the default Kokoro export voice when `--phoneme-render-backend=kokoro`
- `--style-transfer-backend openvoice_v2` applies Coqui OpenVoice v2 after the woman render
- `--style-transfer-target-voice` must match a preset name from `config/style_transfer_presets.json`

## Phoneme Render Backend Comparison

The current production baseline is:

- `coqui_tacotron2_ddc_ph`

Two experimental direct-phoneme comparison backends are exposed:

- `phoneme_vits`
- `fastpitch`
- `kokoro`

Quick probe:

```powershell
python build_blabber_library.py --manifest blabber_manifest.csv --print-backend-probes
```

Run the comparison harness:

```powershell
python compare_phoneme_render_backends.py `
  --manifest blabber_manifest.csv `
  --generation-mode global_soft `
  --output-root comparison-assets
```

The comparison script:

- probes each backend for compatibility or configuration status
- runs the same manifest through each available backend
- prints JSON with probe results, generated outputs, and per-entry failures

Experimental backend constraints:

- `phoneme_vits` requires `BLABBER_PHONEME_VITS_MODEL`
- `fastpitch` requires `nemo_toolkit` plus the NVIDIA FastPitch and HiFiGAN checkpoints
- `kokoro` requires the `kokoro` package, `soundfile`, and the English fallback dependency `espeak-ng`
- the GUI now exposes `kokoro` directly on the woman phoneme path with a dedicated Kokoro voice selector

## Blabber Asset Selection From Template Mismatch

`select_blabber_asset.py` selects a pre-generated male or female Blabber asset for a known target word from the output of either:

- `Template_l2_compare.py`
- `Template_l2_compare_v2.py`

The selector is local-path based and does not hard-code external dataset roots. It expects:

- a saved comparison result payload from `compare_signal_to_prebuilt_template(...)`
- a local JSON asset index, or a generated asset root it can scan
- a requested voice bank: `female`/`woman` or `male`/`man`
- a known target word, or a `label_name` already present in the comparison result
- optional generation-mode preference: `auto`, `global`, or `per_phoneme`

Selection behavior:

- supports both template result shapes
- loads `blabber_asset_index.json` first when present, and otherwise scans `global/<word>/` and `per_phoneme/<word>/`
- uses the comparison `times` and `per_time_l2` arrays as the mismatch timeline
- uses asset `source_alignment` sidecars to estimate phoneme spans
- averages mismatch within each phoneme span
- rounds each phoneme mismatch to the nearest `0.1`
- prefers an exact per-phoneme grade match and otherwise falls back to the nearest available asset

Example:

- if a 2-phoneme word has mismatch `0.5` over the first phoneme span and `0.2` over the second, the selector requests the asset with phoneme grades `[0.5, 0.2]`

Example run:

```powershell
python select_blabber_asset.py `
  --comparison-result compare_go_v2.json `
  --asset-index blabber_asset_index.json `
  --voice female `
  --word go `
  --output-json selected_go_asset.json
```

The selector prints JSON to stdout and can optionally save the same result to `--output-json`.

Expected asset index fields per item:

- `word` or `transcript`
- `voice_mode`
- `phoneme_values`
- `audio_path`
- `sidecar_path` or `json_path`
- `source_phonemes`

The asset index can be either:

- a flat array of items
- the nested summary shape produced by `build_blabber_library.py --print-summary-json`

The selection result includes:

- selected `audio_path`
- selected `sidecar_path`
- requested phoneme grades
- selected asset phoneme grades
- phoneme timing spans and mean mismatch values
- `exact_match`
- selection distance metadata

## Phoneme Distance Reference Export

`export_phoneme_distances.py` generates an offline reference file of phoneme neighbors and substitution candidates ranked from near to far using Apple `ml-acn-embed`.

Examples:

```powershell
python export_phoneme_distances.py --output phoneme_distances.json
python export_phoneme_distances.py --output phoneme_distances.csv
```

The JSON export now includes:

- `ranked_neighbors_near_to_far`
- `ranked_single_neighbors_near_to_far`
- `ranked_candidates_near_to_far`
- `ranked_candidate_entries_near_to_far`

`ranked_candidate_entries_near_to_far` is the field used by deterministic Blabber distance progression. Each entry stores:

- `phones`
- `distance`
- `rank`

Notes:

- ACN phone distance uses `model/embedder-64` by default and auto-extracts it from `model.tgz` when available
- override the ACN model path with `ACN_EMBED_MODEL_DIR`
- regenerate `phoneme_distances.json` after exporter changes if you want live GUI/library behavior to use the newest numeric distances
- regenerate `phoneme_distances.json` if you see `per_phone_reference_warning=1`, because legacy candidate ordering can make per-phoneme ladders less smooth
- working towards hybrid phoneme grapheme --> no, mow, nouh, nowe, nheow 

## Notes And Limitations

- Phoneme identities come from heuristic transcript-based G2P, not a production lexicon.
- Signal-domain phoneme segmentation is estimated from the waveform and transcript, not a full phoneme forced aligner.
- `blabber` depends on a phoneme-capable Coqui model; incompatible TTS environments will fail at render time.
- experimental direct phoneme backend comparison does not change the default production woman render path
- Deterministic distance progression is only as good as the exported `phoneme_distances.json` reference.
- Some demo `.wav` files are kept in the repo for quick testing. Generated outputs and caches are ignored by `.gitignore`.

## Optional CLI Pipeline

The repo also includes a pipeline package under `src/` with a CLI entry point:

```powershell
$env:PYTHONPATH="src"
python -m speech_distortion_pipeline.cli --help
```

## License

Add a license file before publishing if the repo will be shared publicly.

## References

- Apple ACN embedder: https://github.com/apple/ml-acn-embed
- Example voice generator site: https://en.text-to-speech.online/
- Example voice generator site: https://evernote.com/ai-text-to-voice
