"""Tests for grouping and clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drumcut.grouping.cluster import (
    auto_threshold,
    cluster_segments,
    organize_output,
)


class TestAutoThreshold:
    """Tests for automatic threshold computation."""

    def test_basic_threshold(self):
        # Simple distance matrix
        distances = np.array(
            [
                [0.0, 0.1, 0.8],
                [0.1, 0.0, 0.7],
                [0.8, 0.7, 0.0],
            ]
        )
        threshold = auto_threshold(distances)

        assert threshold > 0
        assert threshold < 1.0

    def test_default_percentile(self):
        distances = np.array(
            [
                [0.0, 0.2, 0.4, 0.6],
                [0.2, 0.0, 0.3, 0.5],
                [0.4, 0.3, 0.0, 0.7],
                [0.6, 0.5, 0.7, 0.0],
            ]
        )
        # 30th percentile is the default
        threshold = auto_threshold(distances, percentile=30)

        # Upper triangle values: 0.2, 0.4, 0.6, 0.3, 0.5, 0.7
        # Sorted: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
        # 30th percentile should be around 0.3
        assert 0.2 <= threshold <= 0.5

    def test_all_zeros_returns_default(self):
        distances = np.zeros((3, 3))
        threshold = auto_threshold(distances)

        assert threshold == 0.5  # Default fallback

    def test_custom_percentile(self):
        distances = np.array(
            [
                [0.0, 0.1, 0.5, 0.9],
                [0.1, 0.0, 0.4, 0.8],
                [0.5, 0.4, 0.0, 0.6],
                [0.9, 0.8, 0.6, 0.0],
            ]
        )

        threshold_low = auto_threshold(distances, percentile=10)
        threshold_high = auto_threshold(distances, percentile=90)

        assert threshold_low < threshold_high


class TestClusterSegments:
    """Tests for segment clustering."""

    def test_all_similar_one_group(self):
        # All segments very similar
        distances = np.array(
            [
                [0.0, 0.05, 0.08],
                [0.05, 0.0, 0.06],
                [0.08, 0.06, 0.0],
            ]
        )

        groups = cluster_segments(distances, threshold=0.5)

        assert len(groups) == 1
        assert "A" in groups
        assert sorted(groups["A"]) == [0, 1, 2]

    def test_all_different_separate_groups(self):
        # All segments very different
        distances = np.array(
            [
                [0.0, 0.9, 0.95],
                [0.9, 0.0, 0.85],
                [0.95, 0.85, 0.0],
            ]
        )

        groups = cluster_segments(distances, threshold=0.1)

        assert len(groups) == 3
        assert set(groups.keys()) == {"A", "B", "C"}

    def test_two_clusters(self):
        # Two pairs of similar segments
        distances = np.array(
            [
                [0.0, 0.1, 0.8, 0.85],  # 0 similar to 1
                [0.1, 0.0, 0.82, 0.83],
                [0.8, 0.82, 0.0, 0.12],  # 2 similar to 3
                [0.85, 0.83, 0.12, 0.0],
            ]
        )

        groups = cluster_segments(distances, threshold=0.5)

        assert len(groups) == 2
        # Group A should have 0 and 1, Group B should have 2 and 3
        assert sorted(groups["A"]) == [0, 1]
        assert sorted(groups["B"]) == [2, 3]

    def test_auto_threshold(self):
        distances = np.array(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.85],
                [0.9, 0.85, 0.0],
            ]
        )

        # Should auto-compute threshold
        groups = cluster_segments(distances)

        assert len(groups) >= 1
        # All indices should be assigned
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == [0, 1, 2]

    def test_single_segment(self):
        distances = np.array([[0.0]])

        groups = cluster_segments(distances, threshold=0.5)

        assert len(groups) == 1
        assert groups["A"] == [0]

    def test_labels_are_alphabetical(self):
        # 5 completely different segments
        n = 5
        distances = np.ones((n, n)) * 0.9
        np.fill_diagonal(distances, 0)

        groups = cluster_segments(distances, threshold=0.1)

        assert len(groups) == 5
        expected_labels = ["A", "B", "C", "D", "E"]
        assert list(groups.keys()) == expected_labels


class TestOrganizeOutput:
    """Tests for output organization."""

    def test_creates_group_folders(self, tmp_path: Path):
        # Create fake segment files
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        seg1 = segments_dir / "segment_001.mp4"
        seg2 = segments_dir / "segment_002.mp4"
        seg3 = segments_dir / "segment_003.mp4"

        for seg in [seg1, seg2, seg3]:
            seg.write_text("fake video")

        groups = {"A": [0, 1], "B": [2]}
        output_dir = tmp_path / "output"

        organize_output([seg1, seg2, seg3], groups, output_dir)

        # Check folders created
        assert (output_dir / "A").exists()
        assert (output_dir / "B").exists()

        # Check files copied
        assert (output_dir / "A" / "A1.mp4").exists()
        assert (output_dir / "A" / "A2.mp4").exists()
        assert (output_dir / "B" / "B1.mp4").exists()

    def test_manifest_structure(self, tmp_path: Path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        seg1 = segments_dir / "test.mp4"
        seg1.write_text("fake")

        groups = {"A": [0]}
        output_dir = tmp_path / "output"

        manifest = organize_output([seg1], groups, output_dir)

        assert "groups" in manifest
        assert "A" in manifest["groups"]
        assert "segments" in manifest["groups"]["A"]
        assert len(manifest["groups"]["A"]["segments"]) == 1
        assert manifest["groups"]["A"]["segments"][0]["file"] == "A1.mp4"
        assert manifest["groups"]["A"]["segments"][0]["original_index"] == 0

    def test_preserves_file_extension(self, tmp_path: Path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        seg_mp4 = segments_dir / "video.MP4"
        seg_mov = segments_dir / "video.mov"

        seg_mp4.write_text("fake mp4")
        seg_mov.write_text("fake mov")

        groups = {"A": [0], "B": [1]}
        output_dir = tmp_path / "output"

        organize_output([seg_mp4, seg_mov], groups, output_dir)

        assert (output_dir / "A" / "A1.MP4").exists()
        assert (output_dir / "B" / "B1.mov").exists()

    def test_handles_multiple_takes_per_group(self, tmp_path: Path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        segments = []
        for i in range(5):
            seg = segments_dir / f"seg{i}.mp4"
            seg.write_text(f"fake {i}")
            segments.append(seg)

        groups = {"A": [0, 2, 4], "B": [1, 3]}
        output_dir = tmp_path / "output"

        organize_output(segments, groups, output_dir)

        # Check A has 3 files named A1, A2, A3
        assert (output_dir / "A" / "A1.mp4").exists()
        assert (output_dir / "A" / "A2.mp4").exists()
        assert (output_dir / "A" / "A3.mp4").exists()

        # Check B has 2 files named B1, B2
        assert (output_dir / "B" / "B1.mp4").exists()
        assert (output_dir / "B" / "B2.mp4").exists()
