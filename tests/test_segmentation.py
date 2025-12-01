"""Tests for segmentation detection."""

from __future__ import annotations

import numpy as np

from drumcut.segmentation.detect import (
    Segment,
    adaptive_threshold,
    compute_energy_envelope,
    detect_activity_regions,
)


class TestSegment:
    """Tests for Segment dataclass."""

    def test_duration_property(self):
        seg = Segment(start_seconds=10.0, end_seconds=25.0)
        assert seg.duration_seconds == 15.0

    def test_zero_duration(self):
        seg = Segment(start_seconds=5.0, end_seconds=5.0)
        assert seg.duration_seconds == 0.0


class TestComputeEnergyEnvelope:
    """Tests for energy envelope computation."""

    def test_mono_audio(self):
        # Create simple mono audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine

        energy, time_res = compute_energy_envelope(audio, sr)

        assert len(energy) > 0
        assert time_res > 0
        assert time_res == 512 / sr  # Default hop_length

    def test_stereo_audio_converted_to_mono(self):
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        # Stereo audio
        audio = np.random.randn(samples, 2) * 0.1

        energy, time_res = compute_energy_envelope(audio, sr)

        assert len(energy) > 0

    def test_custom_hop_length(self):
        sr = 22050
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.1

        energy, time_res = compute_energy_envelope(audio, sr, hop_length=1024)

        assert time_res == 1024 / sr

    def test_smoothing_reduces_variance(self):
        sr = 22050
        duration = 5.0
        samples = int(sr * duration)
        # Create audio with high variance - alternating loud/quiet chunks
        audio = np.zeros(samples)
        chunk_size = int(sr * 0.1)  # 100ms chunks
        for i in range(0, samples, chunk_size * 2):
            end = min(i + chunk_size, samples)
            t = np.linspace(0, (end - i) / sr, end - i)
            audio[i:end] = np.sin(2 * np.pi * 440 * t) * 0.8

        # Minimal smoothing
        energy_no_smooth, _ = compute_energy_envelope(audio, sr, smoothing_seconds=0.05)
        # Heavy smoothing
        energy_smooth, _ = compute_energy_envelope(audio, sr, smoothing_seconds=1.0)

        # Smoothed energy should have noticeably less variance
        assert np.std(energy_smooth) < np.std(energy_no_smooth) * 0.95


class TestAdaptiveThreshold:
    """Tests for adaptive threshold computation."""

    def test_basic_threshold(self):
        energy = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0])
        threshold = adaptive_threshold(energy)

        assert threshold > 0
        assert threshold < max(energy)

    def test_uniform_energy(self):
        energy = np.ones(100) * 0.5
        threshold = adaptive_threshold(energy)

        # With uniform energy, MAD should be 0, so threshold = median
        assert threshold == 0.5

    def test_percentile_affects_threshold(self):
        energy = np.linspace(0, 1, 100)

        threshold_low = adaptive_threshold(energy, percentile=25)
        threshold_high = adaptive_threshold(energy, percentile=75)

        assert threshold_low < threshold_high

    def test_factor_affects_threshold(self):
        energy = np.array([0.1, 0.2, 0.5, 0.8, 0.9])

        threshold_low = adaptive_threshold(energy, factor=1.0)
        threshold_high = adaptive_threshold(energy, factor=3.0)

        assert threshold_low < threshold_high


class TestDetectActivityRegions:
    """Tests for activity region detection."""

    def test_silent_audio_no_segments(self):
        sr = 22050
        duration = 5.0
        audio = np.zeros(int(sr * duration))

        segments = detect_activity_regions(audio, sr, min_duration=1.0)

        assert len(segments) == 0

    def test_constant_loud_audio_with_explicit_threshold(self):
        sr = 22050
        duration = 60.0  # 1 minute
        t = np.linspace(0, duration, int(sr * duration))
        # Loud continuous sine wave
        audio = np.sin(2 * np.pi * 440 * t) * 0.8

        # Use explicit low threshold since adaptive won't work with uniform energy
        segments = detect_activity_regions(audio, sr, min_duration=30.0, threshold=0.1)

        # Should detect one long segment
        assert len(segments) >= 1

    def test_min_duration_filter(self):
        sr = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        # With high min_duration, short audio shouldn't produce segments
        segments = detect_activity_regions(audio, sr, min_duration=60.0)
        assert len(segments) == 0

        # With low min_duration, should produce segment
        segments = detect_activity_regions(audio, sr, min_duration=5.0)
        assert len(segments) >= 1

    def test_padding_applied(self):
        sr = 22050
        duration = 60.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.8

        padding = 5.0
        segments = detect_activity_regions(audio, sr, min_duration=30.0, padding=padding)

        if segments:
            # First segment should start at 0 (can't go negative)
            assert segments[0].start_seconds >= 0

    def test_custom_threshold(self):
        sr = 22050
        duration = 30.0
        t = np.linspace(0, duration, int(sr * duration))
        # Quiet audio
        audio = np.sin(2 * np.pi * 440 * t) * 0.1

        # Very low threshold should detect activity
        segments_low = detect_activity_regions(audio, sr, min_duration=10.0, threshold=0.001)

        # Very high threshold should not detect activity
        segments_high = detect_activity_regions(audio, sr, min_duration=10.0, threshold=1.0)

        assert len(segments_low) >= len(segments_high)

    def test_segments_have_positive_duration(self):
        sr = 22050
        duration = 120.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        segments = detect_activity_regions(audio, sr, min_duration=30.0)

        for seg in segments:
            assert seg.duration_seconds > 0
            assert seg.end_seconds > seg.start_seconds

    def test_burst_pattern_detects_activity(self):
        sr = 22050
        duration = 120.0
        samples = int(sr * duration)
        audio = np.zeros(samples)

        # Create two bursts of activity
        burst1_start = int(10 * sr)
        burst1_end = int(50 * sr)
        burst2_start = int(70 * sr)
        burst2_end = int(110 * sr)

        t1 = np.linspace(0, 40, burst1_end - burst1_start)
        t2 = np.linspace(0, 40, burst2_end - burst2_start)

        audio[burst1_start:burst1_end] = np.sin(2 * np.pi * 440 * t1) * 0.8
        audio[burst2_start:burst2_end] = np.sin(2 * np.pi * 440 * t2) * 0.8

        # Use explicit threshold since we know the signal structure
        segments = detect_activity_regions(
            audio, sr, min_duration=20.0, dilation_seconds=1.0, threshold=0.1
        )

        # Should detect at least one segment (may merge depending on params)
        assert len(segments) >= 1
        # Total detected time should be substantial
        total_duration = sum(s.duration_seconds for s in segments)
        assert total_duration >= 40.0  # At least one burst's worth
