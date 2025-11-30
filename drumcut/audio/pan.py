"""Stereo panning utilities."""

from __future__ import annotations

import numpy as np

from drumcut.audio.io import ensure_mono


def pan_mono_to_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """
    Pan mono audio to stereo using constant power pan law.

    Args:
        mono: Mono audio data.
        pan: Pan position from -1.0 (full left) to 1.0 (full right).

    Returns:
        Stereo audio data.
    """
    mono = ensure_mono(mono)

    # Clamp pan to valid range
    pan = max(-1.0, min(1.0, pan))

    # Convert pan (-1 to 1) to angle (0 to pi/2)
    angle = (pan + 1) * np.pi / 4

    # Constant power pan law
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    return np.column_stack([mono * left_gain, mono * right_gain])


def apply_stereo_width(stereo: np.ndarray, width: float = 1.0) -> np.ndarray:
    """
    Adjust stereo width.

    Args:
        stereo: Stereo audio data.
        width: Width factor. 0 = mono, 1 = normal, >1 = wider.

    Returns:
        Adjusted stereo audio.
    """
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("Input must be stereo audio")

    left = stereo[:, 0]
    right = stereo[:, 1]

    mid = (left + right) / 2
    side = (left - right) / 2

    # Adjust side signal
    side = side * width

    return np.column_stack([mid + side, mid - side])
