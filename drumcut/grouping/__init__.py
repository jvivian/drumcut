"""Grouping modules for clustering similar segments."""

from drumcut.grouping.cluster import auto_threshold, cluster_segments
from drumcut.grouping.features import extract_chroma
from drumcut.grouping.similarity import compute_distance_matrix, dtw_distance

__all__ = [
    "extract_chroma",
    "dtw_distance",
    "compute_distance_matrix",
    "cluster_segments",
    "auto_threshold",
]
