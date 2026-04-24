#!/usr/bin/env python3
"""
phone_style_augmentation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script applies a phone‑like augmentation to an input audio file by
mixing the original signal with synthetic noise. The amount of noise
introduced varies over time according to a user‑defined envelope, producing
an effect similar to moving a cross‑fader between the clean signal and a
noisy telephone channel.

Key features
------------
* Supports white noise and pink noise. White noise has a flat power
  spectral density, whereas pink noise has equal energy per octave and
  decreases in power as frequency increases【80753503407418†L55-L58】.
* Accepts a sequence of breakpoints describing how the augmentation
  intensity should change over the course of the clip. Each breakpoint
  specifies a relative position in the audio (0.0 – 1.0) and an
  intensity value (0.0 – 1.0). The intensity is linearly interpolated
  between successive breakpoints to produce a smooth envelope across the
  entire audio.
* Reads and writes standard audio formats using the ``soundfile``
  library. Audio is processed in floating‑point format to maintain
  precision and prevent clipping.
* Generates pink noise based on the FFT scaling algorithm: a random
  white noise signal is converted to the frequency domain, scaled by
  ``1/sqrt(f)`` (with the DC component excluded), and transformed back
  to the time domain. The result is then normalised so that the maximum
  absolute amplitude is 1.0【80753503407418†L79-L105】.

Usage
-----
Run the script from the command line specifying the input file, output
file, noise type and breakpoints. Breakpoints are passed as a single
string where each entry is formatted as ``position:intensity`` and
entries are separated by commas. Positions are floats between 0.0 and
1.0 indicating the relative time in the audio clip, and intensities
are floats between 0.0 and 1.0 indicating how much noise should be
mixed in (0.0 – the original signal only, 1.0 – pure noise).

For example, to gradually increase the amount of white noise from 0 at
the beginning of the clip to 0.7 at the halfway point and then fade
back to 0 near the end:

::

    python phone_style_augmentation.py \
        --input_file path/to/input.wav \
        --output_file path/to/output.wav \
        --noise_type white \
        --breakpoints "0.0:0.0,0.5:0.7,1.0:0.0"

Dependencies
------------
The script relies on `numpy` and `soundfile`. Install them via pip
before running the script:

::

    pip install numpy soundfile scipy

Note that ``scipy`` is only required if you wish to load/write
non‑WAV formats via ``soundfile`` or to customise the pink noise
generation further.
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf


def parse_breakpoints(breakpoints_str: str) -> List[Tuple[float, float]]:
    """Parse a comma‑separated list of ``position:intensity`` pairs.

    Parameters
    ----------
    breakpoints_str:
        A string like ``"0.0:0.0,0.5:0.7,1.0:0.0"``.

    Returns
    -------
    List[Tuple[float, float]]
        A list of (position, intensity) tuples sorted by position.
    """
    breakpoints: List[Tuple[float, float]] = []
    if not breakpoints_str:
        raise ValueError("No breakpoints provided")
    for pair in breakpoints_str.split(','):
        try:
            pos_str, inten_str = pair.split(':')
            position = float(pos_str.strip())
            intensity = float(inten_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Breakpoint '{pair}' is invalid. Use the format 'position:intensity'"
            ) from exc
        if not (0.0 <= position <= 1.0):
            raise ValueError(f"Position {position} out of range [0.0, 1.0]")
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"Intensity {intensity} out of range [0.0, 1.0]")
        breakpoints.append((position, intensity))
    # sort by position in ascending order
    breakpoints.sort(key=lambda x: x[0])
    # ensure first breakpoint at 0 and last at 1; if not, extend
    if breakpoints[0][0] > 0.0:
        breakpoints.insert(0, (0.0, breakpoints[0][1]))
    if breakpoints[-1][0] < 1.0:
        breakpoints.append((1.0, breakpoints[-1][1]))
    return breakpoints


def build_envelope(num_samples: int, breakpoints: List[Tuple[float, float]]) -> np.ndarray:
    """Create an envelope by linearly interpolating between breakpoints.

    The returned array has length ``num_samples`` and values between 0 and
    1 representing the proportion of noise to be mixed into the signal at
    each sample.

    Parameters
    ----------
    num_samples:
        Total number of samples in the audio clip.
    breakpoints:
        List of (position, intensity) tuples as returned by
        :func:`parse_breakpoints`. ``position`` should be in [0.0, 1.0].

    Returns
    -------
    numpy.ndarray
        A one‑dimensional array of length ``num_samples``.
    """
    positions = [int(bp[0] * (num_samples - 1)) for bp in breakpoints]
    intensities = [bp[1] for bp in breakpoints]
    # Use numpy.interp to perform linear interpolation across sample indices
    sample_indices = np.arange(num_samples)
    envelope = np.interp(sample_indices, positions, intensities)
    return envelope


def generate_pink_noise(n_samples: int, sample_rate: int) -> np.ndarray:
    """Generate a mono pink noise signal using an FFT‑based method.

    Pink noise is created by scaling the spectrum of white noise by
    ``1/√f`` and then performing an inverse FFT【80753503407418†L79-L105】.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    sample_rate:
        Sample rate of the audio (used when computing frequency bins).

    Returns
    -------
    numpy.ndarray
        A one‑dimensional array containing pink noise, normalised to a
        maximum absolute value of 1.0.
    """
    # generate white noise
    white = np.random.randn(n_samples)
    # compute the FFT
    white_fft = np.fft.rfft(white)
    # compute frequency bins
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    # create scaling factors: 1/√f, skip DC component (freqs[0])
    scale = np.ones_like(freqs)
    # avoid division by zero for the DC component
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    # apply the scaling
    pink_fft = white_fft * scale
    # inverse FFT back to time domain
    pink = np.fft.irfft(pink_fft, n=n_samples)
    # normalise to [-1, 1]
    max_abs = np.max(np.abs(pink))
    if max_abs > 0:
        pink /= max_abs
    return pink


def generate_noise(
    n_samples: int,
    n_channels: int,
    sample_rate: int,
    noise_type: str,
) -> np.ndarray:
    """Generate multi‑channel noise of a given type and length.

    Parameters
    ----------
    n_samples:
        Number of samples per channel.
    n_channels:
        Number of audio channels (e.g. 1 for mono, 2 for stereo).
    sample_rate:
        Sample rate of the audio.
    noise_type:
        Either ``"white"`` or ``"pink"``.

    Returns
    -------
    numpy.ndarray
        A two‑dimensional array of shape (n_samples, n_channels).
    """
    noise = np.zeros((n_samples, n_channels), dtype=np.float64)
    for ch in range(n_channels):
        if noise_type.lower() == "white":
            # white noise: random samples drawn from a standard normal distribution
            channel_noise = np.random.randn(n_samples)
            # normalise to [-1, 1]
            max_abs = np.max(np.abs(channel_noise))
            if max_abs > 0:
                channel_noise = channel_noise / max_abs
        elif noise_type.lower() == "pink":
            channel_noise = generate_pink_noise(n_samples, sample_rate)
        else:
            raise ValueError(
                f"Unsupported noise_type '{noise_type}'. Use 'white' or 'pink'."
            )
        noise[:, ch] = channel_noise
    return noise


def mix_with_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    """Blend a clean signal with noise according to an envelope.

    For each sample ``i`` and channel ``c``, the output is computed as::

        output[i, c] = (1.0 - envelope[i]) * signal[i, c] + envelope[i] * noise[i, c]

    Thus, ``envelope == 0`` retains the original signal, and ``envelope == 1``
    yields pure noise. Intermediate values linearly interpolate between the two.

    Parameters
    ----------
    signal:
        Array of shape (n_samples, n_channels) containing the original audio.
    noise:
        Array of shape (n_samples, n_channels) containing noise samples.
    envelope:
        One‑dimensional array of length n_samples with values in [0.0, 1.0].

    Returns
    -------
    numpy.ndarray
        The mixed audio signal with the same shape as ``signal``.
    """
    # Ensure envelope has correct length
    if len(envelope) != len(signal):
        raise ValueError(
            f"Envelope length {len(envelope)} does not match signal length {len(signal)}"
        )
    # Expand envelope to match channel dimension for broadcasting
    envelope_matrix = envelope[:, np.newaxis]
    mixed = (1.0 - envelope_matrix) * signal + envelope_matrix * noise
    # Clip to [-1, 1] to avoid clipping when writing back to file
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input audio file. Any format supported by soundfile can be used.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path where the augmented audio will be saved.",
    )
    parser.add_argument(
        "--noise_type",
        choices=["white", "pink"],
        default="white",
        help="Type of noise to use (white or pink). White noise has a flat spectrum while pink noise has equal energy per octave【80753503407418†L55-L58】.",
    )
    parser.add_argument(
        "--breakpoints",
        required=True,
        help=(
            "Comma‑separated list of position:intensity pairs specifying the noise envelope. "
            "Positions (0–1) indicate the relative time in the audio, intensities (0–1) indicate "
            "the mixing level. Example: '0.0:0.0,0.5:0.7,1.0:0.0'."
        ),
    )
    args = parser.parse_args()

    # Read the audio file
    data, sample_rate = sf.read(args.input_file, always_2d=True)
    n_samples, n_channels = data.shape
    # Convert to float64 for processing (soundfile returns float64 by default for floating point audio)
    signal = data.astype(np.float64)

    # Parse breakpoints and build envelope
    breakpoints = parse_breakpoints(args.breakpoints)
    envelope = build_envelope(n_samples, breakpoints)

    # Generate noise of the same length and number of channels
    noise = generate_noise(n_samples, n_channels, sample_rate, args.noise_type)

    # Mix the original signal with noise according to the envelope
    mixed = mix_with_noise(signal, noise, envelope)

    # Write the mixed audio back to disk
    sf.write(args.output_file, mixed, samplerate=sample_rate)
    print(
        f"Augmented audio written to {args.output_file} using {args.noise_type} noise and {len(breakpoints)} breakpoints."
    )


if __name__ == "__main__":
    main()