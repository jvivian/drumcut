"""Per-clip video processing with audio replacement."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from drumcut.audio.align import find_offset


@dataclass
class ClipInfo:
    """Information about a processed clip."""

    path: Path
    offset_seconds: float  # Where this clip starts in the mixed audio
    duration_seconds: float
    correlation_strength: float


def get_video_duration(video_path: Path | str) -> float:
    """Get duration of a video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_audio_segment(
    video_path: Path | str,
    duration: float = 30.0,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Extract first N seconds of audio from video for alignment.

    Args:
        video_path: Path to video file.
        duration: Duration to extract in seconds.
        sample_rate: Output sample rate.

    Returns:
        Tuple of (audio array, sample rate).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",  # mono
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            tmp.name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        audio, sr = librosa.load(tmp.name, sr=sample_rate, mono=True)
    return audio, sr


def align_clip_to_mix(
    video_path: Path | str,
    mixed_audio_path: Path | str,
    alignment_window: float = 30.0,
    sample_rate: int = 16000,
) -> tuple[float, float]:
    """
    Find where a video clip's audio aligns with the mixed audio.

    Args:
        video_path: Path to video file.
        mixed_audio_path: Path to mixed audio file.
        alignment_window: Duration to use for alignment (seconds).
        sample_rate: Sample rate for alignment analysis.

    Returns:
        Tuple of (offset_seconds, correlation_strength).
        offset_seconds is where this clip starts in the mixed audio timeline.
    """
    # Extract audio from video
    clip_audio, sr = extract_audio_segment(video_path, alignment_window, sample_rate)

    # Load corresponding portion of mixed audio
    # We load more than alignment_window to search across the full mix
    mixed_audio, _ = librosa.load(mixed_audio_path, sr=sample_rate, mono=True)

    # Find offset using cross-correlation
    offset_samples, correlation = find_offset(mixed_audio, clip_audio, sr)

    # Convert to seconds
    # Positive offset means clip audio starts AFTER mixed audio start
    offset_seconds = offset_samples / sr

    return offset_seconds, correlation


def process_clip(
    video_path: Path | str,
    mixed_audio_path: Path | str,
    output_path: Path | str,
    offset_seconds: float | None = None,
    alignment_window: float = 30.0,
    audio_codec: str = "aac",
    audio_bitrate: str = "256k",
    verbose: bool = False,
) -> ClipInfo:
    """
    Process a single video clip: replace its audio with mixed audio.

    Args:
        video_path: Path to input video file.
        mixed_audio_path: Path to mixed audio file.
        output_path: Path for output video.
        offset_seconds: Pre-computed offset (None = auto-detect).
        alignment_window: Duration for alignment if auto-detecting.
        audio_codec: Audio codec for output.
        audio_bitrate: Audio bitrate for output.
        verbose: Print ffmpeg output.

    Returns:
        ClipInfo with processing details.
    """
    video_path = Path(video_path)
    mixed_audio_path = Path(mixed_audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get video duration
    video_duration = get_video_duration(video_path)

    # Find alignment if not provided
    if offset_seconds is None:
        offset_seconds, correlation = align_clip_to_mix(
            video_path, mixed_audio_path, alignment_window
        )
    else:
        correlation = 1.0  # Pre-computed, assume good

    # Build ffmpeg command
    # -ss on input for fast seek, then use exact duration
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),  # Video input
        "-ss",
        str(max(0, offset_seconds)),  # Seek in mixed audio
        "-t",
        str(video_duration),  # Duration to extract
        "-i",
        str(mixed_audio_path),  # Audio input
        "-map",
        "0:v",  # Video from first input
        "-map",
        "1:a",  # Audio from second input (after seek)
        "-c:v",
        "copy",  # Copy video stream (fast)
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        "-shortest",  # End when shortest stream ends
        str(output_path),
    ]

    subprocess.run(
        cmd,
        capture_output=not verbose,
        check=True,
    )

    return ClipInfo(
        path=output_path,
        offset_seconds=offset_seconds,
        duration_seconds=video_duration,
        correlation_strength=correlation,
    )


def process_session_clips(
    video_files: list[Path],
    mixed_audio_path: Path | str,
    output_dir: Path | str,
    alignment_window: float = 30.0,
    verbose: bool = False,
) -> list[ClipInfo]:
    """
    Process all clips in a session, replacing audio with mixed version.

    Clips are processed sequentially. Each clip is aligned to the mixed audio
    independently, allowing for gaps or overlaps between GoPro chapters.

    Args:
        video_files: List of video file paths (should be sorted by chapter).
        mixed_audio_path: Path to mixed audio file.
        output_dir: Directory for output files.
        alignment_window: Duration for alignment analysis.
        verbose: Print ffmpeg output.

    Returns:
        List of ClipInfo objects for each processed clip.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video_path in video_files:
        output_path = output_dir / f"processed_{video_path.name}"

        clip_info = process_clip(
            video_path=video_path,
            mixed_audio_path=mixed_audio_path,
            output_path=output_path,
            alignment_window=alignment_window,
            verbose=verbose,
        )
        results.append(clip_info)

    return results
