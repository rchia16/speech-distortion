# Speech Distortion GUI

Desktop GUI and supporting pipeline code for speech augmentation experiments with:

- static noise
- volume dropout
- phone style noise mixing
- blabber phoneme mutation with Coqui TTS

The GUI supports both global quality control and per-phoneme quality control. For signal-domain modes, per-phoneme mode applies augmentation to inferred phoneme segments. For `blabber`, per-phoneme mode compiles a mutated phoneme sequence from the slider values and synthesizes that sequence in one Coqui pass.

## Repo Layout

- `speech_distortion_gui.py`: main desktop GUI
- `create_speech_distortions.py`: older batch augmentation script
- `phone_style_augmentation.py`: reference augmentation logic
- `src/speech_distortion_pipeline/`: reusable pipeline modules
- `configs/`: example pipeline config
- `tests/`: tests for pipeline modules
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

The GUI defaults to the phoneme-capable Coqui model:

```text
tts_models/en/ljspeech/tacotron2-DDC_ph
```

You can override it with:

```powershell
$env:BLABBER_COQUI_MODEL="tts_models/en/ljspeech/tacotron2-DDC_ph"
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
- global mode mutates the word according to a single quality value
- per-phoneme mode mutates each phoneme from its own slider and can insert additional phones at lower quality
- output is synthesized as augmented speech only, without mixing the original signal back in

## Notes And Limitations

- Phoneme identities come from heuristic transcript-based G2P, not a production lexicon.
- Signal-domain phoneme segmentation is estimated from the waveform and transcript, not a full phoneme forced aligner.
- `blabber` depends on a phoneme-capable Coqui model; incompatible TTS environments will fail at render time.
- Some demo `.wav` files are kept in the repo for quick testing. Generated outputs and caches are ignored by `.gitignore`.

## Optional CLI Pipeline

The repo also includes a pipeline package under `src/` with a CLI entry point:

```powershell
$env:PYTHONPATH="src"
python -m speech_distortion_pipeline.cli --help
```

## Preparing For GitHub

Recommended before pushing:

```powershell
git init
git add .
git commit -m "Initial speech distortion GUI"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

The `.gitignore` excludes:

- model caches
- generated outputs
- editor settings
- backup files

## License

Add a license file before publishing if the repo will be shared publicly.
