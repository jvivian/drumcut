"""Audio-video synchronization."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from drumcut.audio.align import find_offset
from drumcut.audio.io import load_audio


def extract_audio_from_video(video_path: Path | str, sr: int = 8000) -> tuple[np.ndarray, int]:
    """
    Extract audio track from video file.

    Args:
        video_path: Path to video file.
        sr: Target sample rate.

    Returns:
        Tuple of (audio data, sample rate).
    """
    video_path = Path(video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            str(tmp_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")

        audio, actual_sr = load_audio(tmp_path, sr=sr)
        return audio, actual_sr

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sync_audio_to_video(
    video_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    search_window: float = 30.0,
) -> dict[str, float]:
    """
    Sync external audio to video using cross-correlation.

    Replaces video's audio track with the external audio, aligned.

    Args:
        video_path: Path to video file.
        audio_path: Path to external audio to sync.
        output_path: Output video path.
        search_window: Maximum seconds to search for offset.

    Returns:
        Dict with sync metadata (offset_seconds, correlation_strength).
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract audio from video
    video_audio, sr = extract_audio_from_video(video_path)

    # Load external audio
    ext_audio, ext_sr = load_audio(audio_path, sr=sr)

    # Find offset
    offset_samples, strength = find_offset(video_audio, ext_audio, sr)
    offset_seconds = offset_samples / sr

    # Warn if large offset
    if abs(offset_seconds) > search_window:
        import warnings

        warnings.warn(
            f"Large sync offset detected: {offset_seconds:.2f}s (search window: {search_window}s)",
            stacklevel=2,
        )

    # Mux with offset
    # Positive offset means external audio starts after video starts
    if offset_seconds >= 0:
        # Delay audio
        audio_filter = f"adelay={int(offset_seconds * 1000)}|{int(offset_seconds * 1000)}"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-af",
            audio_filter,
            str(output_path),
        ]
    else:
        # Trim start of audio
        trim_seconds = abs(offset_seconds)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            str(trim_seconds),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-shortest",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr}")

    return {
        "offset_seconds": offset_seconds,
        "correlation_strength": strength,
    }
