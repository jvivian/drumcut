"""Song export with duration-based categorization."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from drumcut.segmentation.detect import Segment
from drumcut.video.clip import ClipInfo


@dataclass
class SongExport:
    """Information about an exported song."""

    path: Path
    duration_seconds: float
    category: str  # "songs" or "misc"
    source_clips: list[str]  # Which clips contributed to this song
    spans_clips: bool  # Whether song crosses clip boundary


def categorize_segment(segment: Segment, min_song_duration: float = 120.0) -> str:
    """
    Categorize a segment by duration.

    Args:
        segment: The segment to categorize.
        min_song_duration: Minimum duration to be considered a "song" (default 2 min).

    Returns:
        "songs" if >= min_song_duration, "misc" otherwise.
    """
    if segment.duration_seconds >= min_song_duration:
        return "songs"
    return "misc"


def map_segment_to_clips(
    segment: Segment,
    clip_infos: list[ClipInfo],
) -> list[tuple[ClipInfo, float, float]]:
    """
    Map a segment (in mixed audio timeline) to processed clips.

    Returns list of (clip, start_in_clip, end_in_clip) for each clip that
    contains part of this segment.

    Args:
        segment: Segment with start/end in mixed audio timeline.
        clip_infos: List of processed clips with their offsets.

    Returns:
        List of (ClipInfo, start_seconds, end_seconds) tuples.
    """
    mappings = []

    for clip in clip_infos:
        clip_start = clip.offset_seconds
        clip_end = clip.offset_seconds + clip.duration_seconds

        # Check if segment overlaps with this clip
        seg_start = segment.start_seconds
        seg_end = segment.end_seconds

        if seg_end <= clip_start or seg_start >= clip_end:
            # No overlap
            continue

        # Calculate overlap region in clip's local time
        overlap_start = max(seg_start, clip_start)
        overlap_end = min(seg_end, clip_end)

        # Convert to clip-local coordinates
        local_start = overlap_start - clip_start
        local_end = overlap_end - clip_start

        mappings.append((clip, local_start, local_end))

    return mappings


def extract_segment_from_clip(
    clip_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    verbose: bool = False,
) -> Path:
    """
    Extract a segment from a video clip using ffmpeg.

    Args:
        clip_path: Path to input clip.
        output_path: Path for output file.
        start_seconds: Start time in clip.
        end_seconds: End time in clip.
        verbose: Print ffmpeg output.

    Returns:
        Path to extracted segment.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(clip_path),
        "-t",
        str(duration),
        "-c",
        "copy",  # Stream copy for speed
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=not verbose, check=True)
    return output_path


def concat_clips(
    clip_paths: list[Path],
    output_path: Path,
    verbose: bool = False,
) -> Path:
    """
    Concatenate multiple clips into one using ffmpeg concat demuxer.

    Args:
        clip_paths: List of clips to concatenate.
        output_path: Path for output file.
        verbose: Print ffmpeg output.

    Returns:
        Path to concatenated file.
    """
    import tempfile

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create concat file list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in clip_paths:
            f.write(f"file '{path}'\n")
        concat_file = f.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=not verbose, check=True)
    finally:
        Path(concat_file).unlink(missing_ok=True)

    return output_path


def export_songs(
    segments: list[Segment],
    clip_infos: list[ClipInfo],
    output_dir: Path | str,
    min_song_duration: float = 120.0,
    min_misc_duration: float = 30.0,
    verbose: bool = False,
) -> list[SongExport]:
    """
    Export segments as songs, categorized by duration.

    Args:
        segments: List of segments (in mixed audio timeline).
        clip_infos: List of processed clips with their offsets.
        output_dir: Base output directory (will create songs/ and misc/ subdirs).
        min_song_duration: Minimum duration for "songs" category (default 2 min).
        min_misc_duration: Minimum duration to export at all (default 30s).
        verbose: Print ffmpeg output.

    Returns:
        List of SongExport objects.
    """
    output_dir = Path(output_dir)
    songs_dir = output_dir / "songs"
    misc_dir = output_dir / "misc"
    songs_dir.mkdir(parents=True, exist_ok=True)
    misc_dir.mkdir(parents=True, exist_ok=True)

    exports = []
    song_count = 0
    misc_count = 0

    for segment in segments:
        # Skip segments that are too short
        if segment.duration_seconds < min_misc_duration:
            continue

        # Map segment to clips
        clip_mappings = map_segment_to_clips(segment, clip_infos)

        if not clip_mappings:
            continue

        # Categorize
        category = categorize_segment(segment, min_song_duration)
        spans_clips = len(clip_mappings) > 1

        # Determine output path
        if category == "songs":
            song_count += 1
            output_path = songs_dir / f"song_{song_count:03d}.mp4"
        else:
            misc_count += 1
            output_path = misc_dir / f"misc_{misc_count:03d}.mp4"

        # Export
        if spans_clips:
            # Need to extract from multiple clips and concat
            temp_clips = []
            for i, (clip, start, end) in enumerate(clip_mappings):
                temp_path = output_dir / "temp" / f"part_{i:03d}.mp4"
                extract_segment_from_clip(clip.path, temp_path, start, end, verbose)
                temp_clips.append(temp_path)

            concat_clips(temp_clips, output_path, verbose)

            # Cleanup temp clips
            for temp_path in temp_clips:
                temp_path.unlink(missing_ok=True)
        else:
            # Single clip - just extract
            clip, start, end = clip_mappings[0]
            extract_segment_from_clip(clip.path, output_path, start, end, verbose)

        exports.append(
            SongExport(
                path=output_path,
                duration_seconds=segment.duration_seconds,
                category=category,
                source_clips=[m[0].path.name for m in clip_mappings],
                spans_clips=spans_clips,
            )
        )

    # Cleanup temp dir if it exists
    temp_dir = output_dir / "temp"
    if temp_dir.exists():
        temp_dir.rmdir()

    return exports
