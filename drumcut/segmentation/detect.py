"""Activity detection using MIDI track energy analysis."""

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


@dataclass
class Segment:
    """Detected activity segment."""
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def compute_energy_envelope(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    smoothing_seconds: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Compute RMS energy envelope with smoothing.

    Args:
        audio: Audio data.
        sr: Sample rate.
        hop_length: Hop length in samples.
        smoothing_seconds: Window for moving average smoothing.

    Returns:
        Tuple of (energy envelope, time resolution in seconds per frame).
    """
    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Compute RMS energy
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    # Smooth with moving average
    smoothing_frames = int(smoothing_seconds * sr / hop_length)
    if smoothing_frames > 1:
        kernel = np.ones(smoothing_frames) / smoothing_frames
        rms = np.convolve(rms, kernel, mode="same")

    time_resolution = hop_length / sr
    return rms, time_resolution


def adaptive_threshold(energy: np.ndarray, percentile: float = 50, factor: float = 1.5) -> float:
    """
    Compute adaptive threshold using median absolute deviation.

    Args:
        energy: Energy envelope.
        percentile: Percentile for median estimate.
        factor: Multiplier for MAD.

    Returns:
        Threshold value.
    """
    median = np.percentile(energy, percentile)
    mad = np.median(np.abs(energy - median))
    return median + factor * mad


def detect_activity_regions(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_duration: float = 30.0,
    padding: float = 3.0,
    dilation_seconds: float = 3.0,
    erosion_seconds: float = 1.0,
    threshold: float | None = None,
) -> list[Segment]:
    """
    Detect activity regions in audio using energy-based thresholding.

    Args:
        audio: Audio data.
        sr: Sample rate.
        hop_length: Hop length for energy computation.
        min_duration: Minimum segment duration in seconds.
        padding: Padding to add around segments in seconds.
        dilation_seconds: Dilation time for merging nearby regions.
        erosion_seconds: Erosion time for removing small blips.
        threshold: Manual threshold (None = auto).

    Returns:
        List of detected Segment objects.
    """
    # Compute energy
    energy, time_res = compute_energy_envelope(audio, sr, hop_length)

    # Threshold
    if threshold is None:
        threshold = adaptive_threshold(energy)

    # Binary mask
    active = energy > threshold

    # Morphological operations
    dilation_frames = int(dilation_seconds / time_res)
    erosion_frames = int(erosion_seconds / time_res)

    if dilation_frames > 0:
        active = binary_dilation(active, iterations=dilation_frames)
    if erosion_frames > 0:
        active = binary_erosion(active, iterations=erosion_frames)

    # Find contiguous regions
    segments = []
    in_region = False
    region_start = 0

    for i, is_active in enumerate(active):
        if is_active and not in_region:
            region_start = i
            in_region = True
        elif not is_active and in_region:
            start_sec = region_start * time_res
            end_sec = i * time_res
            segments.append((start_sec, end_sec))
            in_region = False

    # Handle region that extends to end
    if in_region:
        start_sec = region_start * time_res
        end_sec = len(active) * time_res
        segments.append((start_sec, end_sec))

    # Filter by minimum duration and add padding
    result = []
    for start, end in segments:
        duration = end - start
        if duration >= min_duration:
            padded_start = max(0, start - padding)
            padded_end = end + padding
            result.append(Segment(start_seconds=padded_start, end_seconds=padded_end))

    return result
