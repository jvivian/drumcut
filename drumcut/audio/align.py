"""Audio alignment using cross-correlation."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from drumcut.audio.io import ensure_mono, resample_if_needed


def find_offset(
    reference: np.ndarray,
    target: np.ndarray,
    sr: int,
    analysis_sr: int = 8000,
) -> tuple[int, float]:
    """
    Find the sample offset between two audio signals using cross-correlation.

    Args:
        reference: Reference audio signal.
        target: Target audio signal to align.
        sr: Sample rate of input signals.
        analysis_sr: Sample rate for correlation analysis (lower = faster).

    Returns:
        Tuple of (offset_samples at original sr, correlation strength).
    """
    # Convert to mono
    ref_mono = ensure_mono(reference)
    tgt_mono = ensure_mono(target)

    # Downsample for faster correlation
    ref_down, _ = resample_if_needed(ref_mono, sr, analysis_sr)
    tgt_down, _ = resample_if_needed(tgt_mono, sr, analysis_sr)

    # Cross-correlate
    correlation = correlate(ref_down, tgt_down, mode="full")

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_value = np.abs(correlation[peak_idx])

    # Convert to offset (negative = target is ahead of reference)
    offset_down = peak_idx - len(tgt_down) + 1

    # Scale offset back to original sample rate
    offset_samples = int(offset_down * (sr / analysis_sr))

    # Normalize correlation strength
    norm_factor = np.sqrt(np.sum(ref_down**2) * np.sum(tgt_down**2))
    if norm_factor > 0:
        correlation_strength = peak_value / norm_factor
    else:
        correlation_strength = 0.0

    return offset_samples, correlation_strength


def align_tracks(
    reference: np.ndarray,
    tracks: list[np.ndarray],
    sr: int,
    max_offset_seconds: float = 1.0,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Align multiple tracks to a reference track.

    Args:
        reference: Reference audio signal.
        tracks: List of tracks to align.
        sr: Sample rate.
        max_offset_seconds: Maximum expected offset (warn if exceeded).

    Returns:
        Tuple of (aligned tracks, offsets in samples).
    """
    max_offset_samples = int(max_offset_seconds * sr)
    aligned = []
    offsets = []

    for track in tracks:
        offset, strength = find_offset(reference, track, sr)

        if abs(offset) > max_offset_samples:
            import warnings
            warnings.warn(
                f"Large offset detected: {offset / sr:.3f}s "
                f"(max expected: {max_offset_seconds}s)"
            )

        offsets.append(offset)

        # Apply offset
        if offset > 0:
            # Target is behind reference - pad start
            if track.ndim == 2:
                padding = np.zeros((offset, track.shape[1]))
            else:
                padding = np.zeros(offset)
            aligned_track = np.concatenate([padding, track])
        elif offset < 0:
            # Target is ahead of reference - trim start
            aligned_track = track[-offset:]
        else:
            aligned_track = track

        aligned.append(aligned_track)

    return aligned, offsets
