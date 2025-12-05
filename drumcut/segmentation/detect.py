"""Activity detection using MIDI track energy analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class SongBoundary:
    """A detected song boundary (silence point across all tracks)."""

    timestamp: float  # seconds
    silence_duration: float  # how long the silence lasts


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


def detect_song_boundaries(
    audio_tracks: list[Path] | list[str] | dict[str, Path | str],
    min_silence_duration: float = 3.0,
    hop_length: int = 512,
    smoothing_seconds: float = 0.5,
    threshold_factor: float = 1.0,
) -> list[SongBoundary]:
    """
    Detect song boundaries by finding silence across ALL audio tracks.

    A song boundary occurs when all tracks are simultaneously silent for at
    least `min_silence_duration` seconds. This catches song transitions even
    when the gap between songs is short (10-20s).

    Args:
        audio_tracks: List of audio file paths, or dict mapping role to path.
        min_silence_duration: Minimum silence duration to consider a boundary.
        hop_length: Hop length for energy computation.
        smoothing_seconds: Smoothing window for energy envelope.
        threshold_factor: Multiplier for adaptive threshold (lower = more sensitive).

    Returns:
        List of SongBoundary objects sorted by timestamp.
    """
    # Normalize input to list of paths
    if isinstance(audio_tracks, dict):
        paths = [Path(p) for p in audio_tracks.values()]
    else:
        paths = [Path(p) for p in audio_tracks]

    if not paths:
        return []

    # Load all tracks and compute energy envelopes
    envelopes = []
    common_sr = None
    common_length = None

    for path in paths:
        audio, sr = librosa.load(path, sr=None, mono=True)

        if common_sr is None:
            common_sr = sr
        elif sr != common_sr:
            # Resample to common rate
            audio = librosa.resample(audio, orig_sr=sr, target_sr=common_sr)

        energy, time_res = compute_energy_envelope(audio, common_sr, hop_length, smoothing_seconds)
        envelopes.append(energy)

        if common_length is None:
            common_length = len(energy)
        else:
            common_length = min(common_length, len(energy))

    # Trim all envelopes to common length
    envelopes = [e[:common_length] for e in envelopes]
    time_res = hop_length / common_sr

    # Compute per-track thresholds and binary activity masks
    activity_masks = []
    for energy in envelopes:
        threshold = adaptive_threshold(energy, factor=threshold_factor)
        active = energy > threshold
        activity_masks.append(active)

    # Find regions where ALL tracks are silent (none are active)
    all_active = np.stack(activity_masks, axis=0)
    any_active = np.any(all_active, axis=0)  # True if ANY track is active
    all_silent = ~any_active  # True only when ALL tracks are silent

    # Find contiguous silence regions
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, is_silent in enumerate(all_silent):
        if is_silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            start_sec = silence_start * time_res
            end_sec = i * time_res
            duration = end_sec - start_sec
            if duration >= min_silence_duration:
                # Use midpoint of silence as boundary timestamp
                midpoint = (start_sec + end_sec) / 2
                silence_regions.append(SongBoundary(timestamp=midpoint, silence_duration=duration))
            in_silence = False

    # Handle silence extending to end
    if in_silence:
        start_sec = silence_start * time_res
        end_sec = len(all_silent) * time_res
        duration = end_sec - start_sec
        if duration >= min_silence_duration:
            midpoint = (start_sec + end_sec) / 2
            silence_regions.append(SongBoundary(timestamp=midpoint, silence_duration=duration))

    return silence_regions


def segments_from_boundaries(
    boundaries: list[SongBoundary],
    total_duration: float,
    min_duration: float = 30.0,
    buffer_seconds: float = 10.0,
) -> list[Segment]:
    """
    Convert song boundaries into segments (the playing regions between boundaries).

    Args:
        boundaries: List of SongBoundary objects.
        total_duration: Total duration of the audio in seconds.
        min_duration: Minimum segment duration to keep.
        buffer_seconds: Buffer to add at start/end of each segment.

    Returns:
        List of Segment objects representing playing regions.
    """
    if not boundaries:
        # No boundaries = one big segment
        return [Segment(start_seconds=0, end_seconds=total_duration)]

    segments = []

    # Sort boundaries by timestamp
    sorted_boundaries = sorted(boundaries, key=lambda b: b.timestamp)

    # First segment: start to first boundary
    if sorted_boundaries[0].timestamp > min_duration:
        start = max(0, 0 - buffer_seconds)  # Can't go negative
        end = sorted_boundaries[0].timestamp + buffer_seconds
        segments.append(Segment(start_seconds=start, end_seconds=end))

    # Middle segments: between consecutive boundaries
    for i in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[i].timestamp
        end = sorted_boundaries[i + 1].timestamp
        duration = end - start

        if duration >= min_duration:
            buffered_start = max(0, start - buffer_seconds)
            buffered_end = end + buffer_seconds
            segments.append(Segment(start_seconds=buffered_start, end_seconds=buffered_end))

    # Last segment: last boundary to end
    last_boundary = sorted_boundaries[-1].timestamp
    if total_duration - last_boundary > min_duration:
        start = max(0, last_boundary - buffer_seconds)
        end = min(total_duration, total_duration + buffer_seconds)
        segments.append(Segment(start_seconds=start, end_seconds=end))

    return segments
