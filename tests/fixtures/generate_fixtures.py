#!/usr/bin/env python3
"""Generate synthetic test fixtures for drumcut tests.

Creates small audio and video files that can be committed to git.
Run this script to regenerate fixtures if needed.
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).parent
SAMPLE_RATE = 44100
DURATION_SECONDS = 5  # Short clips for fast tests


def generate_tone(freq: float, duration: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_drum_pattern(duration: float, sr: int) -> np.ndarray:
    """Generate a simple drum-like pattern with transients."""
    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)

    # Add "kick" hits every 0.5 seconds
    kick_interval = int(sr * 0.5)
    for i in range(0, samples, kick_interval):
        # Exponential decay transient
        decay_len = min(int(sr * 0.1), samples - i)
        t = np.arange(decay_len) / sr
        kick = 0.8 * np.exp(-t * 30) * np.sin(2 * np.pi * 60 * t)
        audio[i:i + decay_len] += kick.astype(np.float32)

    # Add some noise bursts (snare-like)
    snare_interval = int(sr * 1.0)
    for i in range(int(sr * 0.25), samples, snare_interval):
        decay_len = min(int(sr * 0.05), samples - i)
        t = np.arange(decay_len) / sr
        snare = 0.5 * np.exp(-t * 50) * np.random.randn(decay_len)
        audio[i:i + decay_len] += snare.astype(np.float32)

    return np.clip(audio, -1, 1)


def generate_music_pattern(duration: float, sr: int) -> np.ndarray:
    """Generate a simple chord progression pattern."""
    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)

    # C major -> G major progression
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [392.00, 493.88, 587.33],  # G major
    ]

    chord_duration = duration / len(chords)

    for i, chord_freqs in enumerate(chords):
        start = int(i * chord_duration * sr)
        end = int((i + 1) * chord_duration * sr)
        t = np.arange(end - start) / sr

        for freq in chord_freqs:
            audio[start:end] += 0.15 * np.sin(2 * np.pi * freq * t)

    return np.clip(audio, -1, 1).astype(np.float32)


def create_audio_fixtures():
    """Create synthetic audio test files."""
    print("Creating audio fixtures...")

    # 1. Simple test tone (for basic I/O tests)
    tone = generate_tone(440, DURATION_SECONDS, SAMPLE_RATE)
    sf.write(FIXTURES_DIR / "test_tone_440hz.wav", tone, SAMPLE_RATE)
    print(f"  Created test_tone_440hz.wav ({len(tone) / SAMPLE_RATE:.1f}s)")

    # 2. Stereo test file
    left = generate_tone(440, DURATION_SECONDS, SAMPLE_RATE, 0.7)
    right = generate_tone(880, DURATION_SECONDS, SAMPLE_RATE, 0.5)
    stereo = np.column_stack([left, right])
    sf.write(FIXTURES_DIR / "test_stereo.wav", stereo, SAMPLE_RATE)
    print(f"  Created test_stereo.wav ({len(stereo) / SAMPLE_RATE:.1f}s)")

    # 3. Simulated session files (STL, STR, MIDI)
    # Left channel - music pattern panned slightly
    music_l = generate_music_pattern(DURATION_SECONDS, SAMPLE_RATE)
    sf.write(FIXTURES_DIR / "2024-01-01-STL.wav", music_l, SAMPLE_RATE)
    print(f"  Created 2024-01-01-STL.wav")

    # Right channel - same pattern with slight delay (simulates real recording)
    delay_samples = int(0.002 * SAMPLE_RATE)  # 2ms delay
    music_r = np.roll(music_l, delay_samples)
    sf.write(FIXTURES_DIR / "2024-01-01-STR.wav", music_r, SAMPLE_RATE)
    print(f"  Created 2024-01-01-STR.wav")

    # MIDI/Drums track
    drums = generate_drum_pattern(DURATION_SECONDS, SAMPLE_RATE)
    sf.write(FIXTURES_DIR / "2024-01-01-Addictive Drums 2.wav", drums, SAMPLE_RATE)
    print(f"  Created 2024-01-01-Addictive Drums 2.wav")

    # 4. Quiet audio (for normalization tests)
    quiet = tone * 0.01  # -40dB
    sf.write(FIXTURES_DIR / "test_quiet.wav", quiet, SAMPLE_RATE)
    print(f"  Created test_quiet.wav")

    # 5. Loud audio (for limiting tests)
    loud = np.clip(tone * 2, -1, 1)
    sf.write(FIXTURES_DIR / "test_loud.wav", loud, SAMPLE_RATE)
    print(f"  Created test_loud.wav")


def create_video_fixtures():
    """Create synthetic video test files using ffmpeg."""
    print("Creating video fixtures...")

    # Create test pattern videos that simulate GoPro files
    # Using ffmpeg's test sources for small file sizes

    video_duration = 3  # seconds

    # Single chapter file
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={video_duration}:size=640x480:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={video_duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "64k",
        "-shortest",
        str(FIXTURES_DIR / "GOPR0001.MP4"),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"  Created GOPR0001.MP4")

    # Multi-chapter files (session 0002)
    for chapter in range(3):
        if chapter == 0:
            filename = "GOPR0002.MP4"
        else:
            filename = f"GP{chapter:02d}0002.MP4"

        # Vary frequency slightly per chapter to distinguish them
        freq = 440 + chapter * 100

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={video_duration}:size=640x480:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={video_duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k",
            "-shortest",
            str(FIXTURES_DIR / filename),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"  Created {filename}")


def main():
    """Generate all test fixtures."""
    print(f"Generating fixtures in {FIXTURES_DIR}")
    print()

    create_audio_fixtures()
    print()
    create_video_fixtures()
    print()

    # Show total size
    total_size = sum(f.stat().st_size for f in FIXTURES_DIR.glob("*") if f.is_file() and f.suffix in [".wav", ".MP4"])
    print(f"Total fixture size: {total_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
