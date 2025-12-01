"""CLI smoke tests."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from drumcut.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)


class TestVersionCommand:
    """Tests for version command."""

    def test_version_shows_version(self):
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "drumcut version" in result.output


class TestProcessCommand:
    """Tests for process command."""

    def test_no_gopro_files_error(self, tmp_path: Path):
        result = runner.invoke(app, ["process", str(tmp_path)])

        assert result.exit_code == 1
        assert "No GoPro files found" in result.output

    def test_dry_run_shows_plan(self, tmp_path: Path):
        # Create minimal GoPro file
        gopro_file = tmp_path / "GOPR0001.MP4"
        gopro_file.write_bytes(b"fake video")

        result = runner.invoke(app, ["process", str(tmp_path), "--dry-run"])

        assert result.exit_code == 0
        assert "dry run" in result.output
        assert "Merge Video" in result.output
        assert "Mix Audio" in result.output

    def test_session_id_not_found(self, tmp_path: Path):
        gopro_file = tmp_path / "GOPR0001.MP4"
        gopro_file.write_bytes(b"fake video")

        result = runner.invoke(app, ["process", str(tmp_path), "--session-id", "9999", "--dry-run"])

        assert result.exit_code == 1
        assert "not found" in result.output


class TestMergeVideoCommand:
    """Tests for merge-video command."""

    def test_no_gopro_files_error(self, tmp_path: Path):
        result = runner.invoke(app, ["merge-video", str(tmp_path)])

        assert result.exit_code == 1
        assert "No GoPro files found" in result.output


class TestMixCommand:
    """Tests for mix command."""

    def test_no_audio_files_error(self, tmp_path: Path):
        result = runner.invoke(app, ["mix", str(tmp_path)])

        assert result.exit_code == 1
        # Should error due to no audio files


class TestNormalizeCommand:
    """Tests for normalize command."""

    def test_file_not_found(self, tmp_path: Path):
        fake_path = tmp_path / "nonexistent.wav"

        result = runner.invoke(app, ["normalize", str(fake_path)])

        assert result.exit_code != 0


class TestSyncCommand:
    """Tests for sync command."""

    def test_missing_files_error(self, tmp_path: Path):
        fake_video = tmp_path / "video.mp4"
        fake_audio = tmp_path / "audio.wav"

        result = runner.invoke(app, ["sync", str(fake_video), str(fake_audio)])

        # Should fail due to missing files
        assert result.exit_code != 0


class TestSegmentCommand:
    """Tests for segment command."""

    def test_requires_midi_track_or_audio_dir(self, tmp_path: Path):
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        result = runner.invoke(app, ["segment", str(fake_video)])

        assert result.exit_code == 1
        assert "Must provide --midi-track or --audio-dir" in result.output


class TestGroupCommand:
    """Tests for group command."""

    def test_no_segments_error(self, tmp_path: Path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        result = runner.invoke(app, ["group", str(segments_dir)])

        assert result.exit_code == 1
        assert "No video segments found" in result.output


class TestHelpOutput:
    """Tests for help output."""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "drumcut" in result.output
        assert "process" in result.output

    def test_process_help(self):
        result = runner.invoke(app, ["process", "--help"])
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "session_folder" in output.lower()
        assert "--skip-merge" in output

    def test_mix_help(self):
        result = runner.invoke(app, ["mix", "--help"])
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "--left-pan" in output
        assert "--right-pan" in output
