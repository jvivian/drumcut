"""Video segment extraction using ffmpeg."""

import subprocess
from pathlib import Path

from drumcut.segmentation.detect import Segment


def extract_segment(
    input_path: Path | str,
    output_path: Path | str,
    start_seconds: float,
    duration_seconds: float,
    fast_seek: bool = True,
) -> None:
    """
    Extract a segment from a video file.

    Args:
        input_path: Input video path.
        output_path: Output video path.
        start_seconds: Start time in seconds.
        duration_seconds: Duration in seconds.
        fast_seek: Use fast seek (before -i). Slight imprecision but much faster.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fast_seek:
        # -ss before -i for fast seek (may have keyframe imprecision)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_seconds),
            "-i", str(input_path),
            "-t", str(duration_seconds),
            "-c", "copy",
            str(output_path),
        ]
    else:
        # -ss after -i for frame-accurate seek (slower)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ss", str(start_seconds),
            "-t", str(duration_seconds),
            "-c", "copy",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed: {result.stderr}")


def extract_segments(
    input_path: Path | str,
    segments: list[Segment],
    output_dir: Path | str,
    prefix: str = "segment",
) -> list[Path]:
    """
    Extract multiple segments from a video.

    Args:
        input_path: Input video path.
        segments: List of Segment objects to extract.
        output_dir: Output directory for segments.
        prefix: Filename prefix for segments.

    Returns:
        List of paths to extracted segment files.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for i, seg in enumerate(segments, start=1):
        output_path = output_dir / f"{prefix}_{i:03d}.mp4"
        extract_segment(
            input_path,
            output_path,
            seg.start_seconds,
            seg.duration_seconds,
        )
        output_paths.append(output_path)

    return output_paths
