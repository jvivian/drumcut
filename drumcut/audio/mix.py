"""Audio mixing utilities."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from drumcut.audio.align import align_tracks
from drumcut.audio.io import detect_track_roles, load_audio, save_audio
from drumcut.audio.normalize import normalize_audio
from drumcut.audio.pan import pan_mono_to_stereo


def _extract_audio_clip(
    audio_path: Path | str,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    """
    Extract first N seconds of audio using ffmpeg (memory-efficient).

    Args:
        audio_path: Path to audio file.
        duration: Duration to extract in seconds.
        sample_rate: Output sample rate.

    Returns:
        Mono audio array.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-y",
            "-t", str(duration),
            "-i", str(audio_path),
            "-ac", "1",  # mono
            "-ar", str(sample_rate),
            "-f", "wav",
            tmp.name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        audio, _ = librosa.load(tmp.name, sr=sample_rate, mono=True)
    return audio


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
    mixed = sum(t * g for t, g in zip(padded, gains, strict=True))

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
            print(f"  {role}: {path.name} ({shape}, {len(audio) / sr:.1f}s)")

    # Align tracks using MIDI as reference (use mono for correlation)
    offsets_info = {}
    if "midi" in tracks_data and len(tracks_data) > 1:
        reference = ensure_mono(tracks_data["midi"])
        other_roles = [r for r in ["left", "right"] if r in tracks_data]

        if other_roles:
            other_tracks_mono = [ensure_mono(tracks_data[r]) for r in other_roles]
            aligned_mono, offsets = align_tracks(reference, other_tracks_mono, sr)

            # Apply same offset to original (possibly stereo) tracks
            for role, offset in zip(other_roles, offsets, strict=True):
                offsets_info[role] = offset / sr
                if verbose:
                    print(f"  Alignment offset for {role}: {offset / sr * 1000:.1f}ms")

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


def mix_session_ffmpeg(
    session_folder: Path | str,
    output_path: Path | str,
    left_pan: float = -0.5,
    right_pan: float = 0.5,
    target_lufs: float = -14.0,
    alignment_window: float = 30.0,
    verbose: bool = False,
) -> dict[str, object]:
    """
    Mix a session's audio tracks using ffmpeg (memory-efficient).

    Uses ffmpeg for mixing to avoid loading entire audio files into memory.
    Only loads the first `alignment_window` seconds for track alignment.

    Args:
        session_folder: Path to folder containing audio files.
        output_path: Output path for mixed audio.
        left_pan: Pan position for left track (-1 to 1).
        right_pan: Pan position for right track (-1 to 1).
        target_lufs: Target loudness for final mix.
        alignment_window: Seconds to load for alignment (default 30s).
        verbose: Print progress info.

    Returns:
        Dict with mixing metadata.
    """

    session_folder = Path(session_folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = list(session_folder.glob("*.wav")) + list(session_folder.glob("*.WAV"))
    roles = detect_track_roles(audio_files)

    if not roles:
        raise ValueError(f"No audio files found in {session_folder}")

    if verbose:
        print(f"Found tracks: {list(roles.keys())}")

    # Get sample rate from first file
    import soundfile as sf

    info = sf.info(roles[list(roles.keys())[0]])
    sr = info.samplerate

    # Find alignment offsets by loading only first N seconds
    offsets_info = {}
    offsets_samples = {}

    if "midi" in roles and len(roles) > 1:
        # Load only first alignment_window seconds for alignment using ffmpeg
        # This avoids loading entire multi-GB files into memory
        midi_clip = _extract_audio_clip(roles["midi"], alignment_window, sr)

        other_roles = [r for r in ["left", "right"] if r in roles]

        if other_roles:
            other_clips = []
            for role in other_roles:
                clip = _extract_audio_clip(roles[role], alignment_window, sr)
                other_clips.append(clip)

            _, offsets = align_tracks(midi_clip, other_clips, sr)

            for role, offset in zip(other_roles, offsets, strict=True):
                offsets_info[role] = offset / sr
                offsets_samples[role] = offset
                if verbose:
                    print(f"  Alignment offset for {role}: {offset / sr * 1000:.1f}ms")

            del other_clips

        del midi_clip

    # Build ffmpeg command for mixing
    # Convert pan values to ffmpeg pan filter format
    # pan=-0.5 means 75% left, 25% right
    # pan=0.5 means 25% left, 75% right

    inputs = []
    filter_parts = []
    input_idx = 0

    for role in ["left", "right", "midi"]:
        if role not in roles:
            continue

        path = roles[role]
        inputs.extend(["-i", str(path)])

        # Apply offset if needed (adelay in milliseconds)
        offset_ms = int(offsets_samples.get(role, 0) / sr * 1000) if role in offsets_samples else 0

        # Get pan value
        if role == "left":
            pan = left_pan
        elif role == "right":
            pan = right_pan
        else:  # midi
            pan = 0.0

        # Build filter for this track
        # First convert to stereo with pan, then apply delay if needed
        # Pan formula: left_gain = sqrt((1-pan)/2), right_gain = sqrt((1+pan)/2)
        left_gain = np.sqrt((1 - pan) / 2)
        right_gain = np.sqrt((1 + pan) / 2)

        track_filter = f"[{input_idx}:a]aformat=channel_layouts=mono,pan=stereo|c0={left_gain:.4f}*c0|c1={right_gain:.4f}*c0"

        if offset_ms > 0:
            track_filter += f",adelay={offset_ms}|{offset_ms}"
        elif offset_ms < 0:
            trim_sec = abs(offset_ms) / 1000
            track_filter += f",atrim=start={trim_sec:.4f}"

        track_filter += f"[a{input_idx}]"
        filter_parts.append(track_filter)
        input_idx += 1

    # Mix all tracks together
    mix_inputs = "".join(f"[a{i}]" for i in range(input_idx))
    filter_parts.append(f"{mix_inputs}amix=inputs={input_idx}:duration=longest[mixed]")

    # Add loudnorm filter for normalization
    filter_parts.append(f"[mixed]loudnorm=I={target_lufs}:TP=-1.0:LRA=11[out]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ar",
        str(sr),
        str(output_path),
    ]

    if verbose:
        print("Running ffmpeg mix...")
        print(f"  Command: {' '.join(cmd[:10])}...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed: {result.stderr}")

    # Measure output loudness
    loudness_result = _measure_loudness_ffmpeg(output_path)

    if verbose:
        print(f"Output: {output_path}")
        print(f"  LUFS: {loudness_result['output_lufs']:.1f}")
        print(f"  True peak: {loudness_result['true_peak_dbtp']:.1f} dBTP")

    return {
        "tracks": list(roles.keys()),
        "sample_rate": sr,
        "offsets": offsets_info,
        "normalization": loudness_result,
    }


def _measure_loudness_ffmpeg(audio_path: Path | str) -> dict:
    """Measure loudness of audio file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse loudnorm output from stderr
    import json
    import re

    # Find the JSON block in stderr
    json_match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', result.stderr, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "input_lufs": float(data.get("input_i", -24)),
                "output_lufs": float(data.get("input_i", -24)),  # Same since we're measuring
                "true_peak_dbtp": float(data.get("input_tp", -1)),
                "gain_applied_db": 0.0,
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback
    return {
        "input_lufs": -14.0,
        "output_lufs": -14.0,
        "true_peak_dbtp": -1.0,
        "gain_applied_db": 0.0,
    }
