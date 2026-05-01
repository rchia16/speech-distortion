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
$env:BLABBER_COQUI_MODEL="tts_models/en/ljspeech/tacotron2-DDC_ph"
$env:BLABBER_COQUI_CLONE_MODEL="tts_models/multilingual/multi-dataset/xtts_v2"
```

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
- supports optional source voice cloning through Coqui `speaker_wav`
- provides a `Generate Sequence` button to preview the resolved phone sequence before rendering
- caches the generated sequence and reuses it for the next render when inputs have not changed

#### Global Blabber

- global mode uses a deterministic per-word distance ladder built from `phoneme_distances.json`
- each quality value maps to a preset index, and each preset index resolves to the next distinct word-level substitution state by cumulative distance
- rendered metadata includes:
  - `global_preset_index`
  - `global_distance_total`
  - `expansion_used`

#### Per-Phoneme Blabber

- each slider controls only its matching source phoneme
- each phoneme resolves through its own deterministic distance ladder using the same shared reference file
- per-phoneme renders now include:
  - `per_phone_preset_indices`
  - `per_phone_distances`
  - `per_phone_distance_total`
  - `per_phone_sequences`

#### Distance Reference Behavior

- the preferred source of truth is `phoneme_distances.json` with `ranked_candidate_entries_near_to_far`
- each candidate entry includes:
  - replacement phone sequence
  - numeric distance
  - rank
- if numeric entries are missing, the runtime falls back to rank-based synthetic distances derived from candidate order

## Standalone Blabber Library Builder

`build_blabber_library.py` generates standalone Blabber assets from a manifest of input WAVs and single-word transcripts.

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

Example runs:

```powershell
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets
python build_blabber_library.py --manifest blabber_manifest.csv --output-dir speech-assets --use-source-speaker-wav
```

Current behavior:

- default generation mode is `global_soft`, which now means deterministic global distance progression
- global generation produces `GLOBAL_BLABBER_PRESET_COUNT` ordered outputs per word
- per-phoneme deterministic ladders use `PER_PHONEME_BLABBER_PRESET_COUNT`
- per-phoneme generation remains available as the original combinatorial mode
- generated Blabber metadata includes global or per-phone distance information depending on generation mode

Notes:

- `transcript` must currently be a single word
- `phoneme_count` must exactly match the detected source phoneme count for that word
- Coqui must be installed in the configured `BLABBER_CONDA_ENV`
- `--use-source-speaker-wav` uses each manifest input WAV as the Coqui reference voice

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

## Notes And Limitations

- Phoneme identities come from heuristic transcript-based G2P, not a production lexicon.
- Signal-domain phoneme segmentation is estimated from the waveform and transcript, not a full phoneme forced aligner.
- `blabber` depends on a phoneme-capable Coqui model; incompatible TTS environments will fail at render time.
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
