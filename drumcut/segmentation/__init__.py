"""Segmentation modules."""

from drumcut.segmentation.detect import compute_energy_envelope, detect_activity_regions
from drumcut.segmentation.slice import extract_segment, extract_segments

__all__ = [
    "compute_energy_envelope",
    "detect_activity_regions",
    "extract_segment",
    "extract_segments",
]
