# drumcut

A Python CLI tool for automating drum practice video preprocessing. Takes raw GoPro footage and multi-track audio exports from Reaper, and outputs segmented, grouped, production-ready clips.

## Overview

**Problem:** I record 1-2 hour drum practice sessions with a GoPro and a multi-track Reaper setup (rendered MIDI, Left, Right). Getting these into DaVinci for final editing requires tedious preprocessing: normalizing audio, panning, syncing, and manually identifying which segments are actual songs vs. noodling.

`drumcut` automates the entire preprocessing pipeline, delivering a folder of video clips categorized by duration and optionally grouped by song similarity, ready for drag-and-drop into your editor.

## Features

- **Memory-efficient per-clip processing** — processes each GoPro chapter individually, no huge merged files
- **Multi-track audio mixing** — normalizes and mixes L/R/MIDI tracks to -14 LUFS stereo
- **Smart song boundary detection** — uses silence across ALL audio tracks to find song transitions
- **Duration-based categorization** — songs (2+ min) go to `songs/`, shorter clips to `misc/`
- **Cross-clip handling** — automatically concatenates songs that span GoPro chapter boundaries
- **Audio-video sync** — replaces camera audio with mixed track via cross-correlation alignment
- **Optional similarity grouping** — clusters similar songs using DTW audio fingerprinting
- **Rich terminal UI** — beautiful progress display with substeps and progress bars

## Installation

```bash
# Create environment with conda/mamba (includes ffmpeg)
mamba env create -f environment.yml
# or: conda env create -f environment.yml

# Activate environment
conda activate drumcut

# The package is installed in editable mode automatically via environment.yml
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check drumcut tests
ruff format drumcut tests
```

## Quick Start

```bash
# Full pipeline
drumcut process ./session_folder/ --output ./processed/

# With similarity grouping
drumcut process ./session_folder/ --output ./processed/ --group-similar

# Dry run to preview
drumcut process ./session_folder/ --dry-run
```

### Session Folder Structure

```
session_folder/
├── 2025-11-28-Addictive Drums 2.wav   # MIDI/rendered drums track
├── 2025-11-28-STL.wav                  # Stereo Left
├── 2025-11-28-STR.wav                  # Stereo Right
├── GOPR2711.MP4                        # GoPro chapter 0
├── GP012711.MP4                        # GoPro chapter 1
├── GP022711.MP4                        # GoPro chapter 2
└── ...
```

### Input File Conventions

**Audio files** (from Reaper export):
- Format: `YYYY-MM-DD-<track_name>.wav`
- Expected tracks:
  - `*-STL.wav` → Stereo Left
  - `*-STR.wav` → Stereo Right
  - Remaining `.wav` → MIDI/rendered drums (auto-detected)

**Video files** (GoPro):
- `GOPR[SSSS].MP4` → First (or only) chapter of session SSSS
- `GP01[SSSS].MP4` → Chapter 1 continuation
- `GP02[SSSS].MP4` → Chapter 2, etc.

## Pipeline

The `drumcut process` command runs through these steps:

```
1. Mix Audio      → Align and mix L/R/MIDI tracks to stereo WAV
2. Song Boundaries → Detect silence across ALL tracks to find transitions
3. Process Clips  → Align each GoPro clip and replace audio with mix
4. Export Songs   → Extract segments to songs/ (≥2 min) and misc/ (30s-2 min)
5. Group Similar  → (optional) Cluster similar songs by audio fingerprint
```

### Why Per-Clip Processing?

GoPro splits long recordings into ~4GB chapters. Instead of merging everything into one huge file, `drumcut` processes each chapter independently:

- **Memory efficient** — no 10GB+ merged files in memory
- **Fast alignment** — only analyzes first 30 seconds per clip
- **Parallel-friendly** — clips are independent (future enhancement)
- **Cross-clip songs** — rare cases where songs span chapters are automatically detected and concatenated

### Song Boundary Detection

Traditional approaches only look at the MIDI track for silence. But if you finish a song and immediately start the next one (10-20 second gap), you might still be playing something on the mics.

`drumcut` uses **multi-track silence detection** — a boundary is only detected when ALL tracks (MIDI, Left, Right) are simultaneously silent. This catches quick song transitions that single-track approaches miss.

## Output Structure

### Default Output

```
processed/
├── songs/
│   ├── song_001.mp4    # First song (≥2 min)
│   ├── song_002.mp4    # Second song
│   └── song_003.mp4    # Third song
└── misc/
    ├── misc_001.mp4    # Short clip (30s-2 min)
    └── misc_002.mp4
```

### With `--group-similar`

```
processed/
├── songs/
│   ├── A/
│   │   ├── song_001.mp4    # First take of song A
│   │   └── song_003.mp4    # Second take of song A
│   └── B/
│       └── song_002.mp4    # Different song
└── misc/
    └── misc_001.mp4
```

## CLI Reference

### `drumcut process`

Main pipeline command.

```bash
drumcut process SESSION_FOLDER [OPTIONS]

Options:
  -o, --output PATH           Output directory [default: ./processed]
  --min-song-duration INT     Minimum duration for "songs" category (seconds) [default: 120]
  --min-misc-duration INT     Minimum duration to export at all (seconds) [default: 30]
  --buffer FLOAT              Buffer around songs (seconds) [default: 10.0]
  --min-silence FLOAT         Minimum silence for song boundary (seconds) [default: 3.0]
  --session-id INT            Specific GoPro session ID to process
  --skip-mix                  Skip audio mixing (use existing)
  --skip-clips                Skip clip processing (use existing)
  --group-similar             Group similar songs by audio fingerprint
  --keep-intermediates        Keep intermediate files
  --dry-run                   Preview operations without executing
```

### `drumcut mix`

Mix audio tracks only.

```bash
drumcut mix SESSION_FOLDER --output mixed.wav
```

### `drumcut normalize`

Normalize a single audio file.

```bash
drumcut normalize audio.wav --target-lufs -14 --true-peak -1
```

### `drumcut info`

Show information about a session folder.

```bash
drumcut info SESSION_FOLDER
```

## GoPro Naming Convention

GoPro uses different prefixes for chapter files:

| Prefix | Format | Meaning |
|--------|--------|---------|
| `GOPR` | `GOPR[SSSS].MP4` | First (or only) chapter of session SSSS |
| `GP01` | `GP01[SSSS].MP4` | Chapter 1 continuation |
| `GP02` | `GP02[SSSS].MP4` | Chapter 2 continuation |
| `GPnn` | `GPnn[SSSS].MP4` | Chapter n continuation |

**Important:** The session ID is the *last 4 digits*. So `GP012711` means "chapter 01 of session 2711".

**Example ordering:**
```
GOPR2711.MP4  → Session 2711, Chapter 0
GP012711.MP4  → Session 2711, Chapter 1
GP022711.MP4  → Session 2711, Chapter 2
GP032711.MP4  → Session 2711, Chapter 3
```

## Technical Details

### Audio Processing

1. **Alignment** — Uses cross-correlation to align L/R/MIDI tracks (handles slight drift)
2. **Panning** — L track panned left, R track panned right, MIDI centered
3. **Mixing** — Tracks summed and normalized to -14 LUFS with -1 dBTP ceiling
4. **Efficiency** — Only first 30 seconds extracted for alignment analysis

### Video Processing

1. **Per-clip sync** — Each GoPro chapter aligned independently to mixed audio
2. **Stream copy** — Video streams copied without re-encoding (fast)
3. **Audio replacement** — Camera audio replaced with mixed track at correct offset

### Similarity Grouping

When `--group-similar` is enabled:

1. Audio extracted from each exported song (22050 Hz mono)
2. Chroma features computed for each song
3. DTW (Dynamic Time Warping) distance matrix calculated
4. Hierarchical clustering groups similar songs
5. Songs moved to group subdirectories (A/, B/, C/, etc.)

### Dependencies

| Library | Purpose |
|---------|---------|
| `librosa` | Audio analysis, chroma features, DTW |
| `pyloudnorm` | LUFS normalization |
| `soundfile` | Audio I/O |
| `scipy` | Signal processing, clustering |
| `rich` | Terminal UI, progress bars |
| `typer` | CLI framework |
| `ffmpeg` | Video/audio processing (external) |

## Performance

- **8kHz downsampling** for boundary detection (~6x faster than full rate)
- **30-second windows** for clip alignment (no need to analyze full files)
- **Stream copy** for video (no re-encoding)
- **Memory efficient** — processes clips individually, not as one giant file

## Known Limitations

- Assumes single GoPro (no multi-cam sync)
- Assumes 3-track setup (STL, STR, rendered MIDI)
- Audio files must be WAV format
- Audio filenames must follow pattern with STL/STR suffixes

## License

MIT — do whatever you want with it.

## Author

Built to solve a very specific problem: spending less time in DaVinci and more time behind the kit.
