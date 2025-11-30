"""Tests for GoPro filename parsing."""

import pytest

from drumcut.video.gopro import parse_gopro_filename


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
