"""Audio I/O utilities."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: Path | str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        path: Path to audio file.
        sr: Target sample rate. If None, uses native sample rate.

    Returns:
        Tuple of (audio data, sample rate).
    """
    path = Path(path)
    audio, sample_rate = sf.read(path)

    if sr is not None and sr != sample_rate:
        audio = librosa.resample(audio.T, orig_sr=sample_rate, target_sr=sr).T
        sample_rate = sr

    return audio, sample_rate


def save_audio(path: Path | str, audio: np.ndarray, sr: int) -> None:
    """
    Save audio to file.

    Args:
        path: Output path.
        audio: Audio data.
        sr: Sample rate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono if stereo."""
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert audio to stereo if mono."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    return audio


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Resample audio if sample rate doesn't match target."""
    if sr == target_sr:
        return audio, sr

    # Handle stereo
    if audio.ndim == 2:
        resampled = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T
    else:
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return resampled, target_sr


def detect_track_roles(audio_files: list[Path]) -> dict[str, Path]:
    """
    Auto-detect which audio file is Left, Right, and MIDI.

    Convention:
      - *-STL.wav -> Left
      - *-STR.wav -> Right
      - Remaining .wav -> MIDI (rendered drums/backing)

    Args:
        audio_files: List of audio file paths.

    Returns:
        Dict mapping role ('left', 'right', 'midi') to file path.
    """
    roles: dict[str, Path] = {}
    remaining: list[Path] = []

    for f in audio_files:
        stem = f.stem.upper()
        if stem.endswith("-STL"):
            roles["left"] = f
        elif stem.endswith("-STR"):
            roles["right"] = f
        else:
            remaining.append(f)

    if len(remaining) == 1:
        roles["midi"] = remaining[0]
    elif len(remaining) > 1:
        # Multiple candidates - pick the one with "drum" or "midi" in name
        for f in remaining:
            if any(kw in f.stem.lower() for kw in ["drum", "midi", "addictive"]):
                roles["midi"] = f
                break
        else:
            # Fall back to first remaining
            roles["midi"] = remaining[0]

    return roles
