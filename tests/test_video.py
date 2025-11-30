"""Tests for video processing utilities."""

from pathlib import Path

import pytest

from drumcut.video.gopro import (
    GoProFile,
    find_gopro_files,
    group_by_session,
    parse_gopro_filename,
    validate_session_continuity,
)
from drumcut.video.merge import merge_chapters


class TestParseGoProFilename:
    """Tests for parse_gopro_filename function."""

    def test_single_chapter_file(self) -> None:
        """Test parsing single-chapter (GOPR) files."""
        session, chapter = parse_gopro_filename("GOPR2707.MP4")
        assert session == 2707
        assert chapter == 0

    def test_single_chapter_file_lowercase(self) -> None:
        """Test parsing with lowercase extension."""
        session, chapter = parse_gopro_filename("GOPR2708.mp4")
        assert session == 2708
        assert chapter == 0

    def test_multi_chapter_first(self) -> None:
        """Test parsing first chapter of multi-chapter recording."""
        session, chapter = parse_gopro_filename("GOPR2711.MP4")
        assert session == 2711
        assert chapter == 0

    def test_multi_chapter_continuation(self) -> None:
        """Test parsing continuation chapters."""
        session, chapter = parse_gopro_filename("GP012711.MP4")
        assert session == 2711
        assert chapter == 1

        session, chapter = parse_gopro_filename("GP052711.MP4")
        assert session == 2711
        assert chapter == 5

    def test_invalid_format(self) -> None:
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Unknown GoPro filename format"):
            parse_gopro_filename("video.mp4")

        with pytest.raises(ValueError, match="Unknown GoPro filename format"):
            parse_gopro_filename("IMG_1234.MP4")


class TestFindGoProFiles:
    """Tests for find_gopro_files function."""

    def test_finds_all_gopro_files(self, fixtures_dir: Path) -> None:
        """Test finding all GoPro files in a directory."""
        files = find_gopro_files(fixtures_dir)

        # Should find 4 files: GOPR0001, GOPR0002, GP010002, GP020002
        assert len(files) == 4

        # Should be sorted by session, then chapter
        assert files[0].session_id == 1
        assert files[0].chapter == 0
        assert files[1].session_id == 2
        assert files[1].chapter == 0
        assert files[2].session_id == 2
        assert files[2].chapter == 1


class TestGroupBySession:
    """Tests for group_by_session function."""

    def test_groups_correctly(self, fixtures_dir: Path) -> None:
        """Test grouping files by session ID."""
        files = find_gopro_files(fixtures_dir)
        sessions = group_by_session(files)

        assert 1 in sessions
        assert 2 in sessions
        assert len(sessions[1]) == 1  # Single file
        assert len(sessions[2]) == 3  # Multi-chapter


class TestValidateSessionContinuity:
    """Tests for validate_session_continuity function."""

    def test_continuous_session_no_warnings(self) -> None:
        """Test that continuous chapters produce no warnings."""
        files = [
            GoProFile(Path("GOPR0001.MP4"), 1, 0),
            GoProFile(Path("GP010001.MP4"), 1, 1),
            GoProFile(Path("GP020001.MP4"), 1, 2),
        ]
        warnings = validate_session_continuity(files)
        assert len(warnings) == 0

    def test_missing_chapter_produces_warning(self) -> None:
        """Test that missing chapters produce warnings."""
        files = [
            GoProFile(Path("GOPR0001.MP4"), 1, 0),
            GoProFile(Path("GP020001.MP4"), 1, 2),  # Missing chapter 1
        ]
        warnings = validate_session_continuity(files)
        assert len(warnings) == 1
        assert "Missing chapter" in warnings[0]


class TestMergeChapters:
    """Tests for video merging."""

    def test_merge_single_file_copies(
        self, gopro_single_chapter: Path, tmp_video_dir: Path
    ) -> None:
        """Test that merging a single file just copies it."""
        output_path = tmp_video_dir / "merged.mp4"
        files = [GoProFile(gopro_single_chapter, 1, 0)]

        result = merge_chapters(files, output_path)

        assert result.exists()
        assert result.stat().st_size > 0

    def test_merge_multi_chapter(
        self, gopro_multi_chapter: list[Path], tmp_video_dir: Path
    ) -> None:
        """Test merging multiple chapter files."""
        output_path = tmp_video_dir / "merged.mp4"
        files = [
            GoProFile(gopro_multi_chapter[0], 2, 0),
            GoProFile(gopro_multi_chapter[1], 2, 1),
            GoProFile(gopro_multi_chapter[2], 2, 2),
        ]

        result = merge_chapters(files, output_path)

        assert result.exists()
        # Merged file should be larger than individual chapters
        # (roughly 3x, minus any overlap trimming)
        individual_size = sum(p.stat().st_size for p in gopro_multi_chapter)
        # Allow some variance due to container overhead
        assert result.stat().st_size > individual_size * 0.8
