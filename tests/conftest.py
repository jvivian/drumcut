"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock audio file names."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def sample_audio_files(tmp_audio_dir: Path) -> list[Path]:
    """Create sample audio file placeholders."""
    files = [
        tmp_audio_dir / "2024-01-15-STL.wav",
        tmp_audio_dir / "2024-01-15-STR.wav",
        tmp_audio_dir / "2024-01-15-Addictive Drums 2.wav",
    ]
    for f in files:
        f.touch()
    return files
