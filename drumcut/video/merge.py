"""Video chapter merging using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path

from drumcut.video.gopro import GoProFile


def create_concat_file(files: list[GoProFile], output_path: Path) -> None:
    """
    Create an ffmpeg concat demuxer file.

    Args:
        files: List of GoProFile objects in order.
        output_path: Path to write the concat file.
    """
    with open(output_path, "w") as f:
        for gopro_file in files:
            # Escape single quotes in paths
            escaped_path = str(gopro_file.path.absolute()).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")


def merge_chapters(
    files: list[GoProFile],
    output_path: Path | str,
    overlap_trim: float = 0.0,
) -> Path:
    """
    Merge GoPro chapter files into a single video.

    Uses ffmpeg's concat demuxer for fast, lossless merging.

    Args:
        files: List of GoProFile objects to merge, in order.
        output_path: Output video path.
        overlap_trim: Seconds to trim from end of each chapter (except last).

    Returns:
        Path to merged video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(files) == 1:
        # Single file, just copy
        import shutil
        shutil.copy(files[0].path, output_path)
        return output_path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if overlap_trim > 0:
            # Trim overlaps from chapters
            trimmed_files = []
            for i, f in enumerate(files):
                is_last = i == len(files) - 1
                if is_last:
                    # Don't trim the last file
                    trimmed_files.append(f)
                else:
                    # Trim end of this file
                    trimmed_path = tmpdir / f"trimmed_{i:03d}.mp4"
                    _trim_end(f.path, trimmed_path, overlap_trim)
                    trimmed_files.append(
                        GoProFile(path=trimmed_path, session_id=f.session_id, chapter=f.chapter)
                    )
            files = trimmed_files

        # Create concat file
        concat_file = tmpdir / "concat.txt"
        create_concat_file(files, concat_file)

        # Run ffmpeg concat
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    return output_path


def _trim_end(input_path: Path, output_path: Path, seconds: float) -> None:
    """Trim seconds from the end of a video."""
    # First, get duration
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    duration = float(result.stdout.strip())
    new_duration = duration - seconds

    if new_duration <= 0:
        raise ValueError(f"Trim of {seconds}s exceeds video duration {duration}s")

    # Trim
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-t", str(new_duration),
        "-c", "copy",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr}")
