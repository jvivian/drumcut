"""Similarity computation using DTW."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.spatial.distance import cdist

from drumcut.grouping.features import FeatureCache, extract_chroma


def dtw_distance(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """
    Compute DTW distance between two chroma sequences.

    Args:
        chroma1: First chroma feature matrix (n_chroma x time_frames).
        chroma2: Second chroma feature matrix.

    Returns:
        Normalized DTW distance (lower = more similar).
    """
    # Cost matrix using cosine distance
    C = cdist(chroma1.T, chroma2.T, metric="cosine")

    # DTW alignment
    D, _ = librosa.sequence.dtw(C=C, subseq=False)

    # Normalize by path length
    distance = D[-1, -1] / (chroma1.shape[1] + chroma2.shape[1])

    return distance


def compute_distance_matrix(
    audio_paths: list[Path | str],
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """
    Compute pairwise DTW distance matrix for audio files.

    Args:
        audio_paths: List of paths to audio files.
        cache_dir: Optional directory for feature caching.

    Returns:
        Square distance matrix (n_files x n_files).
    """
    n = len(audio_paths)
    audio_paths = [Path(p) for p in audio_paths]

    # Extract features
    if cache_dir:
        cache = FeatureCache(cache_dir)
        chromas = [
            cache.get_or_compute(p, "chroma", extract_chroma)
            for p in audio_paths
        ]
    else:
        chromas = [extract_chroma(p) for p in audio_paths]

    # Compute pairwise distances
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_distance(chromas[i], chromas[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances
