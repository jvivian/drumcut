"""GoPro file naming convention parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoProFile:
    """Parsed GoPro file information."""

    path: Path
    session_id: int
    chapter: int

    @property
    def is_first_chapter(self) -> bool:
        """Return True if this is the first/only chapter of a session."""
        return self.chapter == 0


def parse_gopro_filename(filename: str) -> tuple[int, int]:
    """
    Parse GoPro filename into (session_id, chapter).

    GoPro naming convention:
    - GOPR[SSSS].MP4 = First (or only) chapter of session SSSS
    - GP01[SSSS].MP4 = Chapter 1 continuation
    - GP02[SSSS].MP4 = Chapter 2 continuation
    - etc.

    Args:
        filename: GoPro filename (e.g., "GOPR2711.MP4" or "GP022711.MP4").

    Returns:
        Tuple of (session_id, chapter) where chapter 0 = first/only file.

    Raises:
        ValueError: If filename doesn't match GoPro naming convention.
    """
    # GOPR prefix = chapter 0
    if match := re.match(r"GOPR(\d{4})\.MP4", filename, re.IGNORECASE):
        return (int(match.group(1)), 0)

    # GPnn prefix = chapter n
    if match := re.match(r"GP(\d{2})(\d{4})\.MP4", filename, re.IGNORECASE):
        chapter = int(match.group(1))
        session = int(match.group(2))
        return (session, chapter)

    raise ValueError(f"Unknown GoPro filename format: {filename}")


def find_gopro_files(folder: Path | str) -> list[GoProFile]:
    """
    Find and parse all GoPro files in a folder.

    Args:
        folder: Path to search for GoPro files.

    Returns:
        List of parsed GoProFile objects, sorted by session then chapter.
    """
    folder = Path(folder)
    files = []

    for path in folder.iterdir():
        if path.suffix.upper() == ".MP4":
            try:
                session_id, chapter = parse_gopro_filename(path.name)
                files.append(GoProFile(path=path, session_id=session_id, chapter=chapter))
            except ValueError:
                # Not a GoPro file, skip
                continue

    # Sort by session ID, then chapter
    files.sort(key=lambda f: (f.session_id, f.chapter))
    return files


def group_by_session(files: list[GoProFile]) -> dict[int, list[GoProFile]]:
    """
    Group GoPro files by session ID.

    Args:
        files: List of parsed GoProFile objects.

    Returns:
        Dict mapping session ID to list of files, ordered by chapter.
    """
    sessions: dict[int, list[GoProFile]] = {}

    for f in files:
        if f.session_id not in sessions:
            sessions[f.session_id] = []
        sessions[f.session_id].append(f)

    # Sort each session's files by chapter
    for session_id in sessions:
        sessions[session_id].sort(key=lambda f: f.chapter)

    return sessions


def validate_session_continuity(files: list[GoProFile]) -> list[str]:
    """
    Check for missing chapters in a session.

    Args:
        files: List of files from a single session, sorted by chapter.

    Returns:
        List of warning messages for any gaps detected.
    """
    if not files:
        return []

    warnings = []
    expected_chapter = 0

    for f in files:
        if f.chapter != expected_chapter:
            warnings.append(
                f"Missing chapter(s) in session {f.session_id}: "
                f"expected {expected_chapter}, found {f.chapter}"
            )
        expected_chapter = f.chapter + 1

    return warnings
