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
    verbose: bool = False,
) -> dict[str, object]:
    """
    Mix a session's audio tracks (Left, Right, MIDI).

    Handles both mono and stereo input tracks:
    - Mono tracks are panned to the specified position
    - Stereo tracks are kept as-is (or width-adjusted)

    Args:
        session_folder: Path to folder containing audio files.
        output_path: Output path for mixed audio.
        left_pan: Pan position for left track (-1 to 1).
        right_pan: Pan position for right track (-1 to 1).
        target_lufs: Target loudness for final mix.
        verbose: Print progress info.

    Returns:
        Dict with mixing metadata.
    """
    from drumcut.audio.io import ensure_mono

    session_folder = Path(session_folder)
    output_path = Path(output_path)

    # Find audio files
    audio_files = list(session_folder.glob("*.wav")) + list(session_folder.glob("*.WAV"))
    roles = detect_track_roles(audio_files)

    if not roles:
        raise ValueError(f"No audio files found in {session_folder}")

    if verbose:
        print(f"Found tracks: {list(roles.keys())}")

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
        if verbose:
            shape = "stereo" if audio.ndim == 2 else "mono"
            print(f"  {role}: {path.name} ({shape}, {len(audio)/sr:.1f}s)")

    # Align tracks using MIDI as reference (use mono for correlation)
    offsets_info = {}
    if "midi" in tracks_data and len(tracks_data) > 1:
        reference = ensure_mono(tracks_data["midi"])
        other_roles = [r for r in ["left", "right"] if r in tracks_data]

        if other_roles:
            other_tracks_mono = [ensure_mono(tracks_data[r]) for r in other_roles]
            aligned_mono, offsets = align_tracks(reference, other_tracks_mono, sr)

            # Apply same offset to original (possibly stereo) tracks
            for role, offset in zip(other_roles, offsets):
                offsets_info[role] = offset / sr
                if verbose:
                    print(f"  Alignment offset for {role}: {offset/sr*1000:.1f}ms")

                original = tracks_data[role]
                if offset > 0:
                    # Pad start
                    if original.ndim == 2:
                        padding = np.zeros((offset, original.shape[1]))
                    else:
                        padding = np.zeros(offset)
                    tracks_data[role] = np.concatenate([padding, original])
                elif offset < 0:
                    # Trim start
                    tracks_data[role] = original[-offset:]

    # Process and pan tracks
    processed_tracks = []

    for role in ["left", "right", "midi"]:
        if role not in tracks_data:
            continue

        track = tracks_data[role]

        if track.ndim == 1:
            # Mono track - pan it
            if role == "left":
                processed = pan_mono_to_stereo(track, left_pan)
            elif role == "right":
                processed = pan_mono_to_stereo(track, right_pan)
            else:  # midi
                processed = pan_mono_to_stereo(track, 0.0)  # Center
        else:
            # Stereo track - use as-is (already has stereo positioning)
            processed = track

        processed_tracks.append(processed)

    # Mix
    if verbose:
        print("Mixing tracks...")
    mixed = mix_tracks(processed_tracks)

    # Normalize
    if verbose:
        print("Normalizing...")
    temp_path = output_path.with_suffix(".tmp.wav")
    save_audio(temp_path, mixed, sr)

    result = normalize_audio(temp_path, output_path, target_lufs)
    temp_path.unlink()

    if verbose:
        print(f"Output: {output_path}")
        print(f"  LUFS: {result['output_lufs']:.1f}")
        print(f"  True peak: {result['true_peak_dbtp']:.1f} dBTP")

    return {
        "tracks": list(roles.keys()),
        "sample_rate": sr,
        "offsets": offsets_info,
        "normalization": result,
    }
