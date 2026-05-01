from pathlib import Path


def test_package_layout_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "speech_distortion_pipeline").exists()
    assert (root / "configs" / "pipeline.example.yaml").exists()


def test_first_slice_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "speech_distortion_pipeline" / "config" / "loader.py").exists()
    assert (root / "src" / "speech_distortion_pipeline" / "alignment" / "factory.py").exists()
    assert (root / "src" / "speech_distortion_pipeline" / "cli.py").exists()
