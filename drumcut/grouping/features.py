"""Audio feature extraction for similarity comparison."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import librosa
import numpy as np


def extract_chroma(
    audio_path: Path | str,
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> np.ndarray:
    """
    Extract chroma features from audio file.

    Args:
        audio_path: Path to audio file.
        sr: Sample rate for analysis.
        hop_length: Hop length in samples.
        n_chroma: Number of chroma bins.

    Returns:
        Chroma feature matrix (n_chroma x time_frames).
    """
    y, _ = librosa.load(audio_path, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)

    # Normalize each frame
    chroma = librosa.util.normalize(chroma, axis=0)

    return chroma


def extract_tempo(audio_path: Path | str, sr: int = 22050) -> float:
    """
    Extract tempo (BPM) from audio file.

    Args:
        audio_path: Path to audio file.
        sr: Sample rate for analysis.

    Returns:
        Estimated tempo in BPM.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


class FeatureCache:
    """Simple file-based cache for extracted features."""

    def __init__(self, cache_dir: Path | str):
        """
        Initialize feature cache.

        Args:
            cache_dir: Directory to store cached features.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, audio_path: Path, feature_type: str) -> str:
        """Generate cache key from file path and feature type."""
        path_str = str(Path(audio_path).absolute())
        return hashlib.md5(f"{path_str}:{feature_type}".encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, audio_path: Path | str, feature_type: str) -> np.ndarray | None:
        """
        Get cached features if available.

        Args:
            audio_path: Path to audio file.
            feature_type: Type of feature (e.g., "chroma").

        Returns:
            Cached features or None if not cached.
        """
        key = self._get_cache_key(Path(audio_path), feature_type)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        return None

    def set(self, audio_path: Path | str, feature_type: str, features: np.ndarray) -> None:
        """
        Cache extracted features.

        Args:
            audio_path: Path to audio file.
            feature_type: Type of feature (e.g., "chroma").
            features: Feature array to cache.
        """
        key = self._get_cache_key(Path(audio_path), feature_type)
        cache_path = self._get_cache_path(key)

        with open(cache_path, "wb") as f:
            pickle.dump(features, f)

    def get_or_compute(
        self,
        audio_path: Path | str,
        feature_type: str,
        compute_fn: callable,
    ) -> np.ndarray:
        """
        Get cached features or compute and cache them.

        Args:
            audio_path: Path to audio file.
            feature_type: Type of feature.
            compute_fn: Function to compute features (takes audio_path).

        Returns:
            Feature array.
        """
        cached = self.get(audio_path, feature_type)
        if cached is not None:
            return cached

        features = compute_fn(audio_path)
        self.set(audio_path, feature_type, features)
        return features
