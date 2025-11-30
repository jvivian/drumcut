"""Video chapter merging using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path

from drumcut.audio.align import find_offset
from drumcut.video.gopro import GoProFile


def _extract_audio_segment(
    video_path: Path,
    output_path: Path,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    sr: int = 8000,
) -> None:
    """Extract audio segment from video file."""
    cmd = ["ffmpeg", "-y"]

    if start_seconds is not None:
        cmd.extend(["-ss", str(start_seconds)])

    cmd.extend(["-i", str(video_path)])

    if duration_seconds is not None:
        cmd.extend(["-t", str(duration_seconds)])

    cmd.extend(["-vn", "-ac", "1", "-ar", str(sr), str(output_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")


def _get_video_duration(video_path: Path) -> float:
    """Get duration of video file in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def detect_chapter_overlap(
    chapter1_path: Path,
    chapter2_path: Path,
    window: float = 6.0,
    max_overlap: float = 3.0,
    sr: int = 16000,
) -> tuple[float, float]:
    """
    Detect overlap between two consecutive GoPro chapters using audio correlation.

    Uses normalized cross-correlation on audio from the end of chapter 1 and
    start of chapter 2 to find where they align.

    Args:
        chapter1_path: Path to first (earlier) chapter.
        chapter2_path: Path to second (later) chapter.
        window: Seconds to extract from each chapter for comparison.
        max_overlap: Maximum expected overlap (limits search range).
        sr: Sample rate for audio comparison (higher = more precise).

    Returns:
        Tuple of (overlap_seconds, signal_to_noise_ratio).
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import correlate

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Get duration of chapter 1
        ch1_duration = _get_video_duration(chapter1_path)

        # Extract last N seconds from chapter 1
        ch1_audio_path = tmpdir / "ch1_end.wav"
        _extract_audio_segment(
            chapter1_path,
            ch1_audio_path,
            start_seconds=ch1_duration - window,
            duration_seconds=window,
            sr=sr,
        )

        # Extract first N seconds from chapter 2
        ch2_audio_path = tmpdir / "ch2_start.wav"
        _extract_audio_segment(
            chapter2_path,
            ch2_audio_path,
            duration_seconds=window,
            sr=sr,
        )

        # Load and normalize audio
        ch1_audio, _ = sf.read(ch1_audio_path)
        ch2_audio, _ = sf.read(ch2_audio_path)

        ch1_audio = (ch1_audio - np.mean(ch1_audio)) / (np.std(ch1_audio) + 1e-10)
        ch2_audio = (ch2_audio - np.mean(ch2_audio)) / (np.std(ch2_audio) + 1e-10)

        # Cross-correlate
        corr = correlate(ch1_audio, ch2_audio, mode="full") / len(ch2_audio)
        lags = np.arange(-(len(ch2_audio) - 1), len(ch1_audio))

        # Search only in reasonable overlap range (0 to max_overlap seconds)
        mask = (lags >= 0) & (lags <= max_overlap * sr)
        search_corr = corr[mask]
        search_lags = lags[mask]

        # Find best match
        best_idx = np.argmax(search_corr)
        best_lag = search_lags[best_idx]
        best_corr = search_corr[best_idx]

        # Calculate signal-to-noise ratio (confidence measure)
        snr = (best_corr - np.mean(search_corr)) / (np.std(search_corr) + 1e-10)

        overlap_seconds = best_lag / sr

        return overlap_seconds, snr


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
    overlap_trim: float | str = "auto",
    verbose: bool = False,
) -> Path:
    """
    Merge GoPro chapter files into a single video.

    Uses ffmpeg's concat demuxer for fast, lossless merging.

    Args:
        files: List of GoProFile objects to merge, in order.
        output_path: Output video path.
        overlap_trim: Seconds to trim, or "auto" to detect via audio correlation.
        verbose: Print overlap detection info.

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

        # Detect or use fixed overlap for each chapter boundary
        overlaps: list[float] = []
        for i in range(len(files) - 1):
            if overlap_trim == "auto":
                overlap, strength = detect_chapter_overlap(
                    files[i].path, files[i + 1].path
                )
                if verbose:
                    print(f"  Ch{i}→Ch{i+1}: {overlap:.2f}s overlap (correlation: {strength:.3f})")
                overlaps.append(overlap)
            else:
                overlaps.append(float(overlap_trim))

        # Trim overlaps from chapters
        trimmed_files = []
        for i, f in enumerate(files):
            is_last = i == len(files) - 1
            if is_last:
                # Don't trim the last file
                trimmed_files.append(f)
            else:
                trim_amount = overlaps[i]
                if trim_amount > 0:
                    # Trim end of this file
                    trimmed_path = tmpdir / f"trimmed_{i:03d}.mp4"
                    _trim_end(f.path, trimmed_path, trim_amount)
                    trimmed_files.append(
                        GoProFile(path=trimmed_path, session_id=f.session_id, chapter=f.chapter)
                    )
                else:
                    trimmed_files.append(f)
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
