from pathlib import Path

from speech_distortion_gui import render_pronunciation_safe, subtitle_text_for_mode
from speech_distortion_pipeline.io import WavAudioReader


def test_pronunciation_safe_subtitle_mentions_three_controls() -> None:
    subtitle = subtitle_text_for_mode("pronunciation_safe")
    assert "jibberish" in subtitle.lower()
    assert "clarity" in subtitle.lower()
    assert "timing" in subtitle.lower()


def test_render_pronunciation_safe_returns_audio_with_slider_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    audio = WavAudioReader().read(str(root / "yes_slow.wav"))

    rendered = render_pronunciation_safe(
        audio,
        "string",
        jibberish=0.8,
        clarity=0.5,
        timing_instability=0.7,
    )

    assert rendered.samples
    assert rendered.metadata["augmentation"] == "pronunciation_safe"
    assert rendered.metadata["jibberish"] == "0.800"
    assert rendered.metadata["clarity"] == "0.500"
    assert rendered.metadata["timing_instability"] == "0.700"
    assert float(rendered.metadata["pronunciation_similarity_score"]) >= float(
        rendered.metadata["pronunciation_similarity_threshold"]
    )
