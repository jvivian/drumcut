"""Tests for audio utilities."""

from pathlib import Path

import numpy as np
import pytest

from drumcut.audio.io import (
    detect_track_roles,
    ensure_mono,
    ensure_stereo,
    load_audio,
    save_audio,
)
from drumcut.audio.normalize import measure_loudness, measure_true_peak, normalize_audio
from drumcut.audio.pan import apply_stereo_width, pan_mono_to_stereo


class TestEnsureMono:
    """Tests for ensure_mono function."""

    def test_mono_passthrough(self) -> None:
        """Test that mono audio passes through unchanged."""
        mono = np.array([1.0, 2.0, 3.0])
        result = ensure_mono(mono)
        np.testing.assert_array_equal(result, mono)

    def test_stereo_to_mono(self) -> None:
        """Test stereo to mono conversion."""
        stereo = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
        result = ensure_mono(stereo)
        expected = np.array([2.0, 3.0, 4.0])  # Average of channels
        np.testing.assert_array_equal(result, expected)


class TestEnsureStereo:
    """Tests for ensure_stereo function."""

    def test_stereo_passthrough(self) -> None:
        """Test that stereo audio passes through unchanged."""
        stereo = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ensure_stereo(stereo)
        np.testing.assert_array_equal(result, stereo)

    def test_mono_to_stereo(self) -> None:
        """Test mono to stereo conversion."""
        mono = np.array([1.0, 2.0, 3.0])
        result = ensure_stereo(mono)
        expected = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        np.testing.assert_array_equal(result, expected)


class TestPan:
    """Tests for pan_mono_to_stereo function."""

    def test_center_pan(self) -> None:
        """Test center panning produces equal L/R levels."""
        mono = np.array([1.0, 1.0])
        result = pan_mono_to_stereo(mono, 0.0)
        # At center, both channels should be equal
        np.testing.assert_allclose(result[:, 0], result[:, 1], rtol=1e-7)

    def test_full_left_pan(self) -> None:
        """Test full left panning."""
        mono = np.array([1.0])
        result = pan_mono_to_stereo(mono, -1.0)
        # Full left: left channel = 1.0, right channel = 0.0
        assert result[0, 0] == pytest.approx(1.0, abs=1e-7)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-7)

    def test_full_right_pan(self) -> None:
        """Test full right panning."""
        mono = np.array([1.0])
        result = pan_mono_to_stereo(mono, 1.0)
        # Full right: left channel = 0.0, right channel = 1.0
        assert result[0, 0] == pytest.approx(0.0, abs=1e-7)
        assert result[0, 1] == pytest.approx(1.0, abs=1e-7)

    def test_half_left_pan(self) -> None:
        """Test 50% left panning - both channels should have signal."""
        mono = np.array([1.0])
        result = pan_mono_to_stereo(mono, -0.5)
        # Left should be louder than right
        assert result[0, 0] > result[0, 1]
        # But both should have signal
        assert result[0, 1] > 0


class TestStereoWidth:
    """Tests for apply_stereo_width function."""

    def test_zero_width_is_mono(self) -> None:
        """Test that width=0 produces mono (identical channels)."""
        stereo = np.array([[0.8, 0.2], [0.6, 0.4]])
        result = apply_stereo_width(stereo, 0.0)
        np.testing.assert_allclose(result[:, 0], result[:, 1], rtol=1e-7)

    def test_normal_width(self) -> None:
        """Test that width=1 preserves original stereo."""
        stereo = np.array([[0.8, 0.2], [0.6, 0.4]])
        result = apply_stereo_width(stereo, 1.0)
        np.testing.assert_allclose(result, stereo, rtol=1e-7)


class TestLoadSaveAudio:
    """Tests for audio I/O functions."""

    def test_load_mono_wav(self, test_tone_path: Path) -> None:
        """Test loading a mono WAV file."""
        audio, sr = load_audio(test_tone_path)
        assert sr == 44100
        assert audio.ndim == 1  # Mono
        assert len(audio) == 44100 * 5  # 5 seconds

    def test_load_stereo_wav(self, test_stereo_path: Path) -> None:
        """Test loading a stereo WAV file."""
        audio, sr = load_audio(test_stereo_path)
        assert sr == 44100
        assert audio.ndim == 2  # Stereo
        assert audio.shape[1] == 2

    def test_save_and_reload(self, test_tone_path: Path, tmp_audio_dir: Path) -> None:
        """Test saving and reloading preserves audio."""
        audio, sr = load_audio(test_tone_path)
        output_path = tmp_audio_dir / "test_output.wav"

        save_audio(output_path, audio, sr)

        reloaded, reloaded_sr = load_audio(output_path)
        assert reloaded_sr == sr
        np.testing.assert_allclose(reloaded, audio, rtol=1e-5)


class TestDetectTrackRoles:
    """Tests for track role auto-detection."""

    def test_detect_session_files(self, session_audio_files: dict[str, Path]) -> None:
        """Test detection of STL, STR, and MIDI files."""
        files = list(session_audio_files.values())
        roles = detect_track_roles(files)

        assert "left" in roles
        assert "right" in roles
        assert "midi" in roles
        assert roles["left"].stem.endswith("-STL")
        assert roles["right"].stem.endswith("-STR")
        assert "Addictive Drums" in roles["midi"].stem


class TestNormalization:
    """Tests for audio normalization."""

    def test_measure_loudness(self, test_tone_path: Path) -> None:
        """Test LUFS measurement returns reasonable value."""
        audio, sr = load_audio(test_tone_path)
        lufs = measure_loudness(audio, sr)
        # A -6dB sine wave should be around -9 LUFS
        assert -20 < lufs < 0

    def test_measure_true_peak(self, test_tone_path: Path) -> None:
        """Test true peak measurement."""
        audio, sr = load_audio(test_tone_path)
        peak = measure_true_peak(audio)
        # Our test tone has amplitude 0.5, so peak should be around -6 dB
        assert -10 < peak < 0

    def test_normalize_quiet_audio(
        self, test_quiet_path: Path, tmp_audio_dir: Path
    ) -> None:
        """Test normalizing quiet audio boosts it to target."""
        output_path = tmp_audio_dir / "normalized.wav"
        result = normalize_audio(test_quiet_path, output_path, target_lufs=-14.0)

        # Check gain was applied (quiet audio needs boost)
        assert result["gain_applied_db"] > 0

        # Check output loudness is close to target
        normalized, sr = load_audio(output_path)
        output_lufs = measure_loudness(normalized, sr)
        assert abs(output_lufs - (-14.0)) < 1.0  # Within 1 LUFS of target

    def test_normalize_respects_true_peak(
        self, test_loud_path: Path, tmp_audio_dir: Path
    ) -> None:
        """Test that normalization doesn't exceed true peak limit."""
        output_path = tmp_audio_dir / "normalized.wav"
        result = normalize_audio(
            test_loud_path, output_path, target_lufs=-14.0, true_peak_limit=-1.0
        )

        # Check true peak is below limit
        assert result["true_peak_dbtp"] <= -1.0
