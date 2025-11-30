"""Audio mixing utilities."""

from pathlib import Path

import numpy as np

from drumcut.audio.align import align_tracks
from drumcut.audio.io import detect_track_roles, ensure_stereo, load_audio, save_audio
from drumcut.audio.normalize import measure_loudness, normalize_audio
from drumcut.audio.pan import pan_mono_to_stereo


def mix_tracks(
    tracks: list[np.ndarray],
    gains: list[float] | None = None,
) -> np.ndarray:
    """
    Mix multiple audio tracks.

    Args:
        tracks: List of audio tracks (all must be same length and channels).
        gains: Optional per-track gains in linear scale.

    Returns:
        Mixed audio.
    """
    if not tracks:
        raise ValueError("No tracks to mix")

    if gains is None:
        gains = [1.0] * len(tracks)

    if len(gains) != len(tracks):
        raise ValueError("Number of gains must match number of tracks")

    # Ensure all tracks have same length
    max_len = max(len(t) for t in tracks)
    padded = []
    for track in tracks:
        if len(track) < max_len:
            if track.ndim == 2:
                padding = np.zeros((max_len - len(track), track.shape[1]))
            else:
                padding = np.zeros(max_len - len(track))
            track = np.concatenate([track, padding])
        padded.append(track)

    # Mix with gains
    mixed = sum(t * g for t, g in zip(padded, gains))

    return mixed


def mix_session(
    session_folder: Path | str,
    output_path: Path | str,
    left_pan: float = -0.5,
    right_pan: float = 0.5,
    target_lufs: float = -14.0,
) -> dict[str, object]:
    """
    Mix a session's audio tracks (Left, Right, MIDI).

    Args:
        session_folder: Path to folder containing audio files.
        output_path: Output path for mixed audio.
        left_pan: Pan position for left track.
        right_pan: Pan position for right track.
        target_lufs: Target loudness for final mix.

    Returns:
        Dict with mixing metadata.
    """
    session_folder = Path(session_folder)
    output_path = Path(output_path)

    # Find audio files
    audio_files = list(session_folder.glob("*.wav")) + list(session_folder.glob("*.WAV"))
    roles = detect_track_roles(audio_files)

    if not roles:
        raise ValueError(f"No audio files found in {session_folder}")

    # Load tracks
    tracks_data = {}
    sr = None

    for role, path in roles.items():
        audio, track_sr = load_audio(path)
        if sr is None:
            sr = track_sr
        elif sr != track_sr:
            from drumcut.audio.io import resample_if_needed
            audio, _ = resample_if_needed(audio, track_sr, sr)
        tracks_data[role] = audio

    # Align tracks using MIDI as reference
    if "midi" in tracks_data:
        reference = tracks_data["midi"]
        other_roles = [r for r in ["left", "right"] if r in tracks_data]
        other_tracks = [tracks_data[r] for r in other_roles]

        aligned, offsets = align_tracks(reference, other_tracks, sr)

        for role, track in zip(other_roles, aligned):
            tracks_data[role] = track

    # Pan tracks
    panned_tracks = []

    if "left" in tracks_data:
        panned_tracks.append(pan_mono_to_stereo(tracks_data["left"], left_pan))

    if "right" in tracks_data:
        panned_tracks.append(pan_mono_to_stereo(tracks_data["right"], right_pan))

    if "midi" in tracks_data:
        panned_tracks.append(ensure_stereo(tracks_data["midi"]))

    # Mix
    mixed = mix_tracks(panned_tracks)

    # Normalize
    # Save temp, normalize, then overwrite
    temp_path = output_path.with_suffix(".tmp.wav")
    save_audio(temp_path, mixed, sr)

    from drumcut.audio.normalize import normalize_audio
    result = normalize_audio(temp_path, output_path, target_lufs)
    temp_path.unlink()

    return {
        "tracks": list(roles.keys()),
        "sample_rate": sr,
        "normalization": result,
    }
