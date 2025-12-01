"""Tests for audio mixing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from drumcut.audio.mix import mix_session, mix_tracks


class TestMixTracks:
    """Tests for mix_tracks function."""

    def test_single_track_passthrough(self):
        track = np.random.randn(1000, 2) * 0.5
        mixed = mix_tracks([track])

        np.testing.assert_array_equal(mixed, track)

    def test_two_tracks_summed(self):
        track1 = np.ones((100, 2)) * 0.3
        track2 = np.ones((100, 2)) * 0.2

        mixed = mix_tracks([track1, track2])

        expected = np.ones((100, 2)) * 0.5
        np.testing.assert_array_almost_equal(mixed, expected)

    def test_custom_gains(self):
        track1 = np.ones((100, 2)) * 0.5
        track2 = np.ones((100, 2)) * 0.5

        mixed = mix_tracks([track1, track2], gains=[1.0, 0.5])

        # track1 * 1.0 + track2 * 0.5 = 0.5 + 0.25 = 0.75
        expected = np.ones((100, 2)) * 0.75
        np.testing.assert_array_almost_equal(mixed, expected)

    def test_different_lengths_padded(self):
        track1 = np.ones((100, 2)) * 0.5
        track2 = np.ones((50, 2)) * 0.3

        mixed = mix_tracks([track1, track2])

        assert len(mixed) == 100
        # First 50 samples: 0.5 + 0.3 = 0.8
        np.testing.assert_array_almost_equal(mixed[:50], np.ones((50, 2)) * 0.8)
        # Last 50 samples: 0.5 + 0 = 0.5
        np.testing.assert_array_almost_equal(mixed[50:], np.ones((50, 2)) * 0.5)

    def test_mono_tracks(self):
        track1 = np.ones(100) * 0.3
        track2 = np.ones(100) * 0.2

        mixed = mix_tracks([track1, track2])

        assert mixed.ndim == 1
        np.testing.assert_array_almost_equal(mixed, np.ones(100) * 0.5)

    def test_empty_tracks_raises(self):
        with pytest.raises(ValueError, match="No tracks"):
            mix_tracks([])

    def test_mismatched_gains_raises(self):
        track1 = np.ones((100, 2))
        track2 = np.ones((100, 2))

        with pytest.raises(ValueError, match="gains must match"):
            mix_tracks([track1, track2], gains=[1.0])  # Only 1 gain for 2 tracks


class TestMixSession:
    """Tests for mix_session function."""

    @pytest.fixture
    def session_with_tracks(self, tmp_path: Path):
        """Create a session folder with test audio files."""
        sr = 44100
        duration = 2.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)

        # Create test tracks
        left_audio = np.sin(2 * np.pi * 440 * t) * 0.3
        right_audio = np.sin(2 * np.pi * 880 * t) * 0.3
        midi_audio = np.sin(2 * np.pi * 220 * t) * 0.3

        # Save with recognizable names
        sf.write(tmp_path / "2024-01-01-STL.wav", left_audio, sr)
        sf.write(tmp_path / "2024-01-01-STR.wav", right_audio, sr)
        sf.write(tmp_path / "2024-01-01-Addictive Drums 2.wav", midi_audio, sr)

        return tmp_path, sr

    def test_basic_mix(self, session_with_tracks, tmp_path: Path):
        session_folder, sr = session_with_tracks
        output_path = tmp_path / "output" / "mixed.wav"

        result = mix_session(session_folder, output_path)

        assert output_path.exists()
        assert "tracks" in result
        assert set(result["tracks"]) == {"left", "right", "midi"}
        assert result["sample_rate"] == sr
        assert "normalization" in result

    def test_output_is_stereo(self, session_with_tracks, tmp_path: Path):
        session_folder, sr = session_with_tracks
        output_path = tmp_path / "output" / "mixed.wav"

        mix_session(session_folder, output_path)

        # Load and check
        audio, loaded_sr = sf.read(output_path)
        assert audio.ndim == 2
        assert audio.shape[1] == 2

    def test_no_audio_files_raises(self, tmp_path: Path):
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()
        output_path = tmp_path / "output.wav"

        with pytest.raises(ValueError, match="No audio files"):
            mix_session(empty_folder, output_path)

    def test_creates_output_directory(self, session_with_tracks, tmp_path: Path):
        session_folder, _ = session_with_tracks
        output_path = tmp_path / "nested" / "deep" / "mixed.wav"

        mix_session(session_folder, output_path)

        assert output_path.exists()

    def test_normalization_result(self, session_with_tracks, tmp_path: Path):
        session_folder, _ = session_with_tracks
        output_path = tmp_path / "mixed.wav"

        result = mix_session(session_folder, output_path, target_lufs=-14.0)

        norm = result["normalization"]
        assert "output_lufs" in norm
        assert "true_peak_dbtp" in norm
        # Output should be close to target (within a few dB)
        assert -20.0 <= norm["output_lufs"] <= -10.0

    def test_single_track_session(self, tmp_path: Path):
        """Test mixing with only one track."""
        sr = 44100
        duration = 2.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)

        # Only MIDI track
        midi_audio = np.sin(2 * np.pi * 220 * t) * 0.3
        sf.write(tmp_path / "2024-01-01-Addictive Drums 2.wav", midi_audio, sr)

        output_path = tmp_path / "mixed.wav"
        result = mix_session(tmp_path, output_path)

        assert output_path.exists()
        assert result["tracks"] == ["midi"]
