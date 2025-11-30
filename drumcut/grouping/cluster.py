"""Clustering algorithms for grouping similar segments."""

from __future__ import annotations

import numpy as np


def auto_threshold(distances: np.ndarray, percentile: float = 30) -> float:
    """
    Automatically determine clustering threshold.

    Uses percentile of non-zero distances as threshold.

    Args:
        distances: Square distance matrix.
        percentile: Percentile to use as threshold.

    Returns:
        Threshold value.
    """
    # Get upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(len(distances), k=1)
    flat_distances = distances[triu_indices]

    # Filter out zeros
    nonzero = flat_distances[flat_distances > 0]

    if len(nonzero) == 0:
        return 0.5  # Default if no valid distances

    return float(np.percentile(nonzero, percentile))


def cluster_segments(
    distances: np.ndarray,
    threshold: float | None = None,
) -> dict[str, list[int]]:
    """
    Cluster segments based on pairwise distances.

    Uses greedy single-linkage clustering.

    Args:
        distances: Square distance matrix.
        threshold: Distance threshold for grouping. None = auto.

    Returns:
        Dict mapping group label ('A', 'B', etc.) to list of segment indices.
    """
    if threshold is None:
        threshold = auto_threshold(distances)

    n = len(distances)
    labeled = [False] * n
    groups: dict[str, list[int]] = {}
    current_label = "A"

    for i in range(n):
        if labeled[i]:
            continue

        # Start new group with segment i
        group = [i]
        labeled[i] = True

        # Find all unlabeled segments similar to i
        for j in range(i + 1, n):
            if not labeled[j] and distances[i, j] < threshold:
                group.append(j)
                labeled[j] = True

        groups[current_label] = group
        current_label = chr(ord(current_label) + 1)

    return groups


def organize_output(
    segment_paths: list,
    groups: dict[str, list[int]],
    output_dir,
) -> dict:
    """
    Organize segments into group folders.

    Args:
        segment_paths: List of segment file paths.
        groups: Dict mapping group label to segment indices.
        output_dir: Output directory.

    Returns:
        Manifest dict with group information.
    """
    from pathlib import Path
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"groups": {}}

    for label, indices in groups.items():
        group_dir = output_dir / label
        group_dir.mkdir(exist_ok=True)

        group_info = {"segments": []}

        for take_num, idx in enumerate(indices, start=1):
            src_path = Path(segment_paths[idx])
            dst_name = f"{label}{take_num}{src_path.suffix}"
            dst_path = group_dir / dst_name

            shutil.copy(src_path, dst_path)

            group_info["segments"].append({
                "file": dst_name,
                "original_index": idx,
            })

        manifest["groups"][label] = group_info

    return manifest
