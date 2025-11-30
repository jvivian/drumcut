"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def test_tone_path(fixtures_dir: Path) -> Path:
    """Path to 440Hz test tone WAV file."""
    return fixtures_dir / "test_tone_440hz.wav"


@pytest.fixture
def test_stereo_path(fixtures_dir: Path) -> Path:
    """Path to stereo test WAV file."""
    return fixtures_dir / "test_stereo.wav"


@pytest.fixture
def test_quiet_path(fixtures_dir: Path) -> Path:
    """Path to quiet (-40dB) test WAV file."""
    return fixtures_dir / "test_quiet.wav"


@pytest.fixture
def test_loud_path(fixtures_dir: Path) -> Path:
    """Path to loud/clipped test WAV file."""
    return fixtures_dir / "test_loud.wav"


@pytest.fixture
def session_audio_files(fixtures_dir: Path) -> dict[str, Path]:
    """Paths to simulated session audio files (STL, STR, MIDI)."""
    return {
        "left": fixtures_dir / "2024-01-01-STL.wav",
        "right": fixtures_dir / "2024-01-01-STR.wav",
        "midi": fixtures_dir / "2024-01-01-Addictive Drums 2.wav",
    }


@pytest.fixture
def gopro_single_chapter(fixtures_dir: Path) -> Path:
    """Path to single-chapter GoPro test video."""
    return fixtures_dir / "GOPR0001.MP4"


@pytest.fixture
def gopro_multi_chapter(fixtures_dir: Path) -> list[Path]:
    """Paths to multi-chapter GoPro test videos (session 0002)."""
    return [
        fixtures_dir / "GOPR0002.MP4",
        fixtures_dir / "GP010002.MP4",
        fixtures_dir / "GP020002.MP4",
    ]


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for audio output."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def tmp_video_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for video output."""
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    return video_dir
