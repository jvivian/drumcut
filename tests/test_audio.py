"""Tests for audio utilities."""

import numpy as np
import pytest

from drumcut.audio.io import detect_track_roles, ensure_mono, ensure_stereo
from drumcut.audio.pan import pan_mono_to_stereo


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
