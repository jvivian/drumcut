"""Segmentation modules."""

from drumcut.segmentation.detect import (
    Segment,
    SongBoundary,
    compute_energy_envelope,
    detect_activity_regions,
    detect_song_boundaries,
    segments_from_boundaries,
)
from drumcut.segmentation.slice import extract_segment, extract_segments

__all__ = [
    "Segment",
    "SongBoundary",
    "compute_energy_envelope",
    "detect_activity_regions",
    "detect_song_boundaries",
    "extract_segment",
    "extract_segments",
    "segments_from_boundaries",
]
