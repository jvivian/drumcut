"""Audio normalization using LUFS measurement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from drumcut.audio.io import load_audio, save_audio


def measure_loudness(audio: np.ndarray, sr: int) -> float:
    """
    Measure integrated loudness in LUFS.

    Args:
        audio: Audio data.
        sr: Sample rate.

    Returns:
        Loudness in LUFS.
    """
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(audio)


def measure_true_peak(audio: np.ndarray) -> float:
    """
    Measure true peak in dBTP.

    Args:
        audio: Audio data.

    Returns:
        True peak in dBTP.
    """
    peak = np.abs(audio).max()
    if peak == 0:
        return -np.inf
    return 20 * np.log10(peak)


def normalize_audio(
    input_path: Path | str,
    output_path: Path | str,
    target_lufs: float = -14.0,
    true_peak_limit: float = -1.0,
) -> dict[str, float]:
    """
    Normalize audio to target LUFS with true peak limiting.

    Args:
        input_path: Input audio file path.
        output_path: Output audio file path.
        target_lufs: Target integrated loudness in LUFS.
        true_peak_limit: True peak ceiling in dBTP.

    Returns:
        Dict with measured values (input_lufs, output_lufs, gain_applied, true_peak).
    """
    audio, sr = load_audio(input_path)

    # Measure input loudness
    input_lufs = measure_loudness(audio, sr)

    # Calculate required gain
    gain_db = target_lufs - input_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain
    normalized = audio * gain_linear

    # Check and limit true peak
    current_peak = measure_true_peak(normalized)
    if current_peak > true_peak_limit:
        # Reduce gain to meet true peak limit
        reduction_db = current_peak - true_peak_limit
        reduction_linear = 10 ** (-reduction_db / 20)
        normalized = normalized * reduction_linear
        gain_db -= reduction_db

    # Final measurements
    output_lufs = measure_loudness(normalized, sr)
    final_peak = measure_true_peak(normalized)

    # Save
    save_audio(output_path, normalized, sr)

    return {
        "input_lufs": input_lufs,
        "output_lufs": output_lufs,
        "gain_applied_db": gain_db,
        "true_peak_dbtp": final_peak,
    }
