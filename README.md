# drumcut

A Python CLI tool for automating drum practice video preprocessing. Takes raw GoPro footage and multi-track audio exports from Reaper, and outputs segmented, grouped, production-ready clips.

## Overview

**Problem:** I record 1-2 hour drum practice sessions with a GoPro and a multi-track Reaper setup (rendered MIDI, Left, Right). Getting these into DaVinci for final editing requires tedious preprocessing: normalizing audio, panning, syncing, converting to greyscale, and manually identifying which segments are actual songs vs. noodling or random noise.

**The Solution:** `drumcut` automates the entire preprocessing pipeline, delivering a folder of greyscale video clips grouped by song similarity, ready for drag-and-drop into your editor.

## Features

- **Audio normalization** to YouTube-optimized -14 LUFS
- **Stereo panning** of L/R tracks with center MIDI
- **Automatic audio alignment** via cross-correlation (handles slight track drift)
- **GoPro chapter merging** with overlap handling
- **Audio-video sync** by correlating camera audio with mixed track
- **Greyscale/filter presets** (death metal aesthetic out of the box)
- **Song segmentation** using MIDI track energy detection
- **Clip grouping** via chroma-based similarity (DTW) — groups multiple takes of the same song

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

# Type checking
mypy drumcut
```

## Quick Start

```bash
# Full pipeline
drumcut process ./session_folder/ --output ./processed/

# Session folder structure expected (flat directory, all files together):
# session_folder/
# ├── 2025-11-28-Addictive Drums 2.wav   # MIDI/rendered drums track
# ├── 2025-11-28-STL.wav                  # Stereo Left
# ├── 2025-11-28-STR.wav                  # Stereo Right
# ├── GOPR2707.MP4                        # GoPro single-chapter clips
# ├── GOPR2708.MP4
# ├── GOPR2709.MP4
# ├── GOPR2710.MP4
# ├── GOPR2711.MP4                        # First chapter of multi-chapter recording
# ├── GP012711.MP4                        # Chapter 1 continuation of 2711
# ├── GP022711.MP4                        # Chapter 2 of 2711
# ├── GP032711.MP4                        # Chapter 3 of 2711
# ├── GP042711.MP4                        # Chapter 4 of 2711
# ├── GP052711.MP4                        # Chapter 5 of 2711
# ├── GOPR2712.MP4                        # First chapter of next recording
# └── GP012712.MP4                        # Chapter 1 continuation of 2712
```

### Input File Conventions

**Audio files** (from Reaper export):
- Format: `YYYY-MM-DD-<track_name>.wav`
- Expected tracks:
  - `*-Addictive Drums 2.wav` or similar → MIDI/rendered drums (auto-detected by exclusion)
  - `*-STL.wav` → Stereo Left
  - `*-STR.wav` → Stereo Right
- The tool auto-detects tracks by suffix: `STL` = left, `STR` = right, remaining = MIDI

**Video files** (GoPro):
- See [GoPro Naming Convention](#gopro-naming-convention) section below

### GoPro Naming Convention

GoPro's file naming is notoriously confusing. Here's how it works:

**Single-chapter recordings** (short clips < ~4GB):
```
GOPR2707.MP4   # Session 2707, complete file
GOPR2708.MP4   # Session 2708, complete file
```

**Multi-chapter recordings** (long recordings split at ~4GB):
```
GOPR2711.MP4   # Session 2711, Chapter 0 (the "head")
GP012711.MP4   # Session 2711, Chapter 1
GP022711.MP4   # Session 2711, Chapter 2
GP032711.MP4   # Session 2711, Chapter 3
...
```

**The pattern:**
| Prefix | Format | Meaning |
|--------|--------|---------|
| `GOPR` | `GOPR[SSSS].MP4` | First (or only) chapter of session SSSS |
| `GP01` | `GP01[SSSS].MP4` | Chapter 1 continuation |
| `GP02` | `GP02[SSSS].MP4` | Chapter 2 continuation |
| `GPnn` | `GPnn[SSSS].MP4` | Chapter n continuation |

**Important:** The session ID is the *last 4 digits*, not the first. So `GP012711` means "chapter 01 of session 2711", not "session GP01 chapter 2711".

**Correct ordering for your example files:**
```
1. GOPR2707.MP4  (session 2707 - single)
2. GOPR2708.MP4  (session 2708 - single)
3. GOPR2709.MP4  (session 2709 - single)
4. GOPR2710.MP4  (session 2710 - single)
5. GOPR2711.MP4  (session 2711, ch 0)
6. GP012711.MP4  (session 2711, ch 1)
7. GP022711.MP4  (session 2711, ch 2)
8. GP032711.MP4  (session 2711, ch 3)
9. GP042711.MP4  (session 2711, ch 4)
10. GP052711.MP4 (session 2711, ch 5)
11. GOPR2712.MP4 (session 2712, ch 0)
12. GP012712.MP4 (session 2712, ch 1)
```

## Output Structure

```
processed/
├── full_session.mp4          # Complete merged/synced video (if requested)
├── segments/
│   ├── A/
│   │   ├── A1.mp4            # First take of song A
│   │   └── A2.mp4            # Second take of song A
│   ├── B/
│   │   └── B1.mp4            # Single take of song B
│   └── C/
│       ├── C1.mp4
│       ├── C2.mp4
│       └── C3.mp4
└── manifest.json             # Metadata: timestamps, groupings, similarity scores
```

---

## Development Plan

### Phase 1: Project Scaffolding & Audio Pipeline

#### Task 1.1: Project Structure Setup
- [ ] Initialize package structure with `pyproject.toml`
- [ ] Set up Typer CLI skeleton in `cli.py`
- [ ] Create logging configuration (we'll want verbose mode for debugging)
- [ ] Add basic config file support (YAML) for default parameters

**Package structure:**
```
drumcut/
├── pyproject.toml
├── README.md
├── drumcut/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── normalize.py
│   │   ├── pan.py
│   │   ├── align.py
│   │   └── mix.py
│   ├── video/
│   │   ├── __init__.py
│   │   ├── gopro.py
│   │   ├── merge.py
│   │   ├── filters.py
│   │   └── sync.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── detect.py
│   │   └── slice.py
│   └── grouping/
│       ├── __init__.py
│       ├── features.py
│       ├── similarity.py
│       └── cluster.py
└── tests/
    └── ...
```

**Dependencies for pyproject.toml:**
```toml
dependencies = [
    "typer",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "pyloudnorm",
    "ffmpeg-python",
    "pyyaml",
    "rich",  # nice progress bars and console output
]
```

#### Task 1.2: Audio I/O Utilities
- [ ] Create `audio/io.py` with functions to load/save audio files
- [ ] Handle WAV file loading via `soundfile` (native support, no conversion needed)
- [ ] Implement stereo/mono detection and conversion
- [ ] Add sample rate validation/resampling (ensure all tracks match)
- [ ] Create track auto-detection from filenames

**Key functions:**
- `load_audio(path) -> (np.ndarray, sample_rate)`
- `save_audio(path, data, sample_rate)`
- `ensure_mono(audio) -> np.ndarray`
- `ensure_stereo(audio) -> np.ndarray`
- `resample_if_needed(audio, sr, target_sr) -> np.ndarray`

**Track auto-detection:**
```python
def detect_track_roles(audio_files: list[Path]) -> dict[str, Path]:
    """
    Auto-detect which audio file is Left, Right, and MIDI.
    
    Convention:
      - *-STL.wav → Left
      - *-STR.wav → Right  
      - Remaining .wav → MIDI (rendered drums/backing)
    """
    roles = {}
    remaining = []
    
    for f in audio_files:
        stem = f.stem.upper()
        if stem.endswith('-STL'):
            roles['left'] = f
        elif stem.endswith('-STR'):
            roles['right'] = f
        else:
            remaining.append(f)
    
    if len(remaining) == 1:
        roles['midi'] = remaining[0]
    elif len(remaining) > 1:
        # Multiple candidates - pick the one with "drum" or "midi" in name
        for f in remaining:
            if any(kw in f.stem.lower() for kw in ['drum', 'midi', 'addictive']):
                roles['midi'] = f
                break
        else:
            # Fall back to first remaining
            roles['midi'] = remaining[0]
    
    return roles
```

#### Task 1.3: Audio Normalization
- [ ] Implement LUFS measurement using `pyloudnorm`
- [ ] Implement true peak measurement
- [ ] Create normalization function targeting -14 LUFS / -1 dBTP
- [ ] Add CLI command: `drumcut normalize`

**Algorithm:**
1. Load audio file
2. Measure integrated LUFS
3. Calculate gain needed to reach -14 LUFS
4. Check if gain would cause true peak to exceed -1 dBTP
5. If so, reduce gain to meet true peak ceiling
6. Apply gain
7. Save normalized audio

**CLI interface:**
```bash
drumcut normalize ./audio/Left.mp3 --target-lufs -14 --true-peak -1 --output ./normalized/
```

#### Task 1.4: Stereo Panning
- [ ] Implement pan law (constant power recommended for music)
- [ ] Create panning function that takes mono input and pan position (-1 to +1)
- [ ] Add CLI command: `drumcut pan`

**Pan law (constant power):**
```python
def pan_mono_to_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """
    Pan mono audio to stereo.
    pan: -1.0 = full left, 0.0 = center, 1.0 = full right
    """
    # Convert pan (-1 to 1) to angle (0 to pi/2)
    angle = (pan + 1) * np.pi / 4
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    return np.column_stack([mono * left_gain, mono * right_gain])
```

**For your setup:**
- Left track: `pan = -0.5` (50% left)
- Right track: `pan = +0.5` (50% right)
- MIDI track: `pan = 0.0` (center)

#### Task 1.5: Audio Alignment
- [ ] Implement cross-correlation based alignment
- [ ] Add downsampling for performance (8kHz is plenty for finding offsets)
- [ ] Handle negative and positive offsets (padding vs trimming)
- [ ] Add CLI command: `drumcut align`

**Algorithm:**
1. Load all three tracks
2. Downsample to 8kHz for correlation computation
3. Use MIDI track as reference (it's the most "clean" signal)
4. Cross-correlate Left with MIDI, find peak offset
5. Cross-correlate Right with MIDI, find peak offset
6. Pad/trim Left and Right to align with MIDI
7. Report offsets found (for logging/debugging)

**Key considerations:**
- Offsets should be small (< 100ms typically) — flag if larger
- Use `scipy.signal.correlate` with `mode='full'`
- Find peak with `np.argmax`, convert to sample offset

#### Task 1.6: Audio Mixing
- [ ] Implement track combination (sum with optional per-track gain)
- [ ] Create mixing function that takes multiple stereo tracks
- [ ] Add headroom management (prevent clipping on sum)
- [ ] Add CLI command: `drumcut mix`

**Mixing approach:**
1. Normalize each track (from Task 1.3)
2. Pan each track (from Task 1.4)
3. Align tracks (from Task 1.5)
4. Sum all tracks: `mixed = left_panned + right_panned + midi_panned`
5. Apply makeup gain if needed to hit -14 LUFS on mix
6. Limit/clip protection

**CLI interface:**
```bash
drumcut mix --left "./2025-11-28-STL.wav" --right "./2025-11-28-STR.wav" \
    --midi "./2025-11-28-Addictive Drums 2.wav" \
    --left-pan -0.5 --right-pan 0.5 --output ./mixed.wav

# Or with auto-detection (recommended):
drumcut mix ./session_folder/ --output ./mixed.wav
```

---

### Phase 2: Video Pipeline

#### Task 2.1: GoPro File Parser
- [ ] Implement filename parser for GoPro naming convention
- [ ] Extract chapter number and session ID from filenames
- [ ] Sort files by session, then chapter
- [ ] Validate continuity (detect missing chapters)
- [ ] Fall back to creation timestamp if naming is ambiguous

**GoPro naming convention:**

GoPro uses different prefixes depending on whether it's the first chapter or a continuation:

| Prefix | Meaning | Format |
|--------|---------|--------|
| `GOPR` | First chapter (or single file) | `GOPR[SSSS].MP4` |
| `GP01` | Chapter 1 continuation | `GP01[SSSS].MP4` |
| `GP02` | Chapter 2 | `GP02[SSSS].MP4` |
| `GPnn` | Chapter n | `GPnn[SSSS].MP4` |

Where `SSSS` = session ID (e.g., 2711)

**Example session with your files:**
```
Session 2711 (multi-chapter, ~25 min recording that split into 6 files):
  GOPR2711.MP4  → Chapter 0 (first ~4 min)
  GP012711.MP4  → Chapter 1
  GP022711.MP4  → Chapter 2
  GP032711.MP4  → Chapter 3
  GP042711.MP4  → Chapter 4
  GP052711.MP4  → Chapter 5

Session 2712 (multi-chapter):
  GOPR2712.MP4  → Chapter 0
  GP012712.MP4  → Chapter 1

Sessions 2707-2710 (single chapter each, short recordings):
  GOPR2707.MP4  → Complete file
  GOPR2708.MP4  → Complete file
  GOPR2709.MP4  → Complete file
  GOPR2710.MP4  → Complete file
```

**Parsing logic:**
```python
import re

def parse_gopro_filename(filename: str) -> tuple[int, int]:
    """
    Parse GoPro filename into (session_id, chapter).
    
    Returns:
        (session_id, chapter) where chapter 0 = first/only file
    """
    # GOPR prefix = chapter 0
    if match := re.match(r'GOPR(\d{4})\.MP4', filename, re.IGNORECASE):
        return (int(match.group(1)), 0)
    
    # GPnn prefix = chapter n
    if match := re.match(r'GP(\d{2})(\d{4})\.MP4', filename, re.IGNORECASE):
        chapter = int(match.group(1))
        session = int(match.group(2))
        return (session, chapter)
    
    raise ValueError(f"Unknown GoPro filename format: {filename}")

# Sorting: group by session, then order by chapter
files_sorted = sorted(gopro_files, key=lambda f: parse_gopro_filename(f.name))
```

**Output:** Ordered list of file paths, grouped by session, ready for concatenation

#### Task 2.2: Video Concatenation
- [ ] Implement ffmpeg concat demuxer approach (no re-encode)
- [ ] Handle chapter overlap trimming (configurable, default 0.5s)
- [ ] Generate concat file list for ffmpeg
- [ ] Add CLI command: `drumcut merge-video`

**ffmpeg approach:**
```bash
# Generate file list
echo "file 'GH010042.MP4'" > list.txt
echo "file 'GH020042.MP4'" >> list.txt
echo "file 'GH030042.MP4'" >> list.txt

# Concat without re-encode
ffmpeg -f concat -safe 0 -i list.txt -c copy merged.mp4
```

**Overlap handling options:**
1. **Trim end of each chapter** (except last): Remove last N seconds before concat
2. **Trim start of each chapter** (except first): Remove first N seconds before concat
3. **Smart detection**: Hash final/initial frames, find exact overlap point

Start with option 1, it's simplest and usually sufficient.

#### Task 2.3: Video-Audio Synchronization
- [ ] Extract audio track from merged video using ffmpeg
- [ ] Cross-correlate extracted audio with mixed audio track
- [ ] Calculate offset (video audio is usually delayed vs. direct recording)
- [ ] Mux new audio with video at correct offset
- [ ] Add CLI command: `drumcut sync`

**Algorithm:**
1. Extract audio: `ffmpeg -i merged.mp4 -vn -ac 1 -ar 8000 video_audio.wav`
2. Load mixed audio, downsample to 8kHz mono
3. Cross-correlate to find offset
4. Mux with offset: `ffmpeg -i merged.mp4 -i mixed.wav -c:v copy -map 0:v -map 1:a -ss {offset} output.mp4`

**Edge cases:**
- If mixed audio is shorter than video: trim video or pad audio
- If mixed audio is longer: warn user (probably stopped GoPro early)
- Large offsets (> 5 seconds): warn, might indicate wrong files

#### Task 2.4: Video Filters
- [ ] Implement filter preset system
- [ ] Create "death-metal" preset (greyscale + contrast + subtle grain)
- [ ] Create "clean" preset (just greyscale)
- [ ] Create "raw" preset (pass-through)
- [ ] Allow custom ffmpeg filter strings
- [ ] Add CLI command: `drumcut filter`

**Preset definitions:**
```python
FILTER_PRESETS = {
    "death-metal": "hue=s=0,eq=contrast=1.2:brightness=-0.03,noise=c0s=8:c0f=t",
    "clean": "hue=s=0",
    "raw": "null",  # no-op filter
    "film": "hue=s=0,curves=m='0/0 0.1/0.05 0.9/0.95 1/1',vignette=PI/5",
}
```

**CLI interface:**
```bash
drumcut filter ./synced.mp4 --preset death-metal --output ./filtered.mp4
drumcut filter ./synced.mp4 --custom "hue=s=0.3,eq=gamma=1.1" --output ./filtered.mp4
```

---

### Phase 3: Segmentation

#### Task 3.1: MIDI Track Energy Analysis
- [ ] Implement RMS energy computation in windows
- [ ] Implement onset strength as alternative/complement
- [ ] Create energy envelope smoothing
- [ ] Add visualization output for debugging (optional matplotlib)

**Approach:**
```python
def compute_energy_envelope(audio: np.ndarray, sr: int, 
                            hop_length: int = 512) -> np.ndarray:
    """Compute RMS energy envelope."""
    return librosa.feature.rms(y=audio, hop_length=hop_length)[0]
```

**Window size considerations:**
- 512 samples @ 44.1kHz ≈ 11.6ms per frame
- Smooth with moving average over ~1 second for stability

#### Task 3.2: Activity Detection
- [ ] Implement adaptive thresholding (handle varying session volumes)
- [ ] Apply morphological operations to clean up detections
- [ ] Filter by minimum duration (< 30s probably not a full song)
- [ ] Add configurable padding before/after segments

**Algorithm:**
1. Compute energy envelope
2. Calculate threshold (e.g., median + 1.5 * MAD, or percentile-based)
3. Create binary mask: `active = energy > threshold`
4. Dilate to merge small gaps (e.g., 3 second dilation)
5. Erode to remove tiny blips (e.g., 1 second erosion)
6. Find contiguous regions
7. Filter: keep only regions > min_duration
8. Expand each region by padding

**Morphological ops in numpy:**
```python
from scipy.ndimage import binary_dilation, binary_erosion

# Convert time to frames
dilation_frames = int(3.0 * sr / hop_length)
erosion_frames = int(1.0 * sr / hop_length)

dilated = binary_dilation(active, iterations=dilation_frames)
cleaned = binary_erosion(dilated, iterations=erosion_frames)
```

#### Task 3.3: Segment Extraction
- [ ] Convert frame indices to timestamps
- [ ] Use ffmpeg stream copy for fast extraction (no re-encode)
- [ ] Generate segment metadata (start time, end time, duration)
- [ ] Add CLI command: `drumcut segment`

**ffmpeg extraction:**
```bash
ffmpeg -ss {start} -i input.mp4 -t {duration} -c copy segment_001.mp4
```

**Note:** `-ss` before `-i` is fast but may have keyframe alignment issues. If precision matters, use `-ss` after `-i` (slower but frame-accurate). For our use case, slight imprecision is fine.

**CLI interface:**
```bash
drumcut segment ./filtered.mp4 --midi-track "./2025-11-28-Addictive Drums 2.wav" \
    --min-duration 30 --padding 3 --output ./segments/

# Or with auto-detection:
drumcut segment ./filtered.mp4 --audio-dir ./session_folder/ \
    --min-duration 30 --padding 3 --output ./segments/
```

---

### Phase 4: Grouping

#### Task 4.1: Feature Extraction
- [ ] Implement chroma feature extraction using librosa
- [ ] Add feature caching (these are expensive to compute)
- [ ] Normalize chroma features for comparison
- [ ] Consider adding tempo feature as secondary signal

**Chroma extraction:**
```python
def extract_chroma(audio_path: str) -> np.ndarray:
    """Extract chroma features from audio."""
    y, sr = librosa.load(audio_path, sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Normalize each frame
    chroma = librosa.util.normalize(chroma, axis=0)
    return chroma
```

**Why chroma works for this:**
- Captures harmonic/tonal content
- Invariant to timbre (your drums vs. backing track)
- Somewhat robust to tempo variation (especially with DTW)
- Two takes of same song → similar chroma profiles

#### Task 4.2: Similarity Computation
- [ ] Implement DTW distance computation
- [ ] Add early termination for obvious non-matches (performance)
- [ ] Create pairwise distance matrix for all segments
- [ ] Cache distance matrix

**DTW implementation:**
```python
from scipy.spatial.distance import cdist

def dtw_distance(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Compute DTW distance between two chroma sequences."""
    # Cost matrix
    C = cdist(chroma1.T, chroma2.T, metric='cosine')
    
    # DTW
    D, wp = librosa.sequence.dtw(C=C, subseq=False)
    
    # Normalized distance
    return D[-1, -1] / (len(chroma1.T) + len(chroma2.T))
```

**Performance note:** For 1-2 hour sessions, you might have 10-20 segments. Pairwise DTW on 20 segments is 190 comparisons — totally tractable. For longer sessions, consider `fastdtw` or hierarchical pruning.

#### Task 4.3: Clustering Algorithm
- [ ] Implement greedy single-linkage clustering (your recursive algorithm)
- [ ] Add automatic threshold detection (bimodal distribution split)
- [ ] Allow manual threshold override
- [ ] Generate group labels (A, B, C, ...)

**Algorithm implementation:**
```python
def cluster_segments(distances: np.ndarray, threshold: float) -> dict[str, list[int]]:
    """
    Cluster segments based on pairwise distances.
    Returns mapping of group label to segment indices.
    """
    n = len(distances)
    labeled = [False] * n
    groups = {}
    current_label = 'A'
    
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
```

**Threshold auto-detection:**
```python
def auto_threshold(distances: np.ndarray) -> float:
    """Find threshold at valley between same-song and different-song distributions."""
    # Flatten upper triangle
    triu_indices = np.triu_indices(len(distances), k=1)
    flat_distances = distances[triu_indices]
    
    # Simple approach: use Otsu's method or find histogram valley
    # For now, use percentile heuristic
    return np.percentile(flat_distances, 30)
```

#### Task 4.4: Output Organization
- [ ] Create output directory structure (group folders)
- [ ] Rename/copy segments with group-aware naming (A1, A2, B1, etc.)
- [ ] Generate manifest.json with full metadata
- [ ] Add CLI command: `drumcut group`

**Output structure:**
```
segments/
├── A/
│   ├── A1.mp4
│   └── A2.mp4
├── B/
│   └── B1.mp4
└── manifest.json
```

**manifest.json schema:**
```json
{
  "session": "2024-01-15",
  "total_duration": "01:45:23",
  "groups": {
    "A": {
      "segments": [
        {"file": "A1.mp4", "start": "00:05:23", "end": "00:09:47", "duration": "00:04:24"},
        {"file": "A2.mp4", "start": "01:12:05", "end": "01:16:33", "duration": "00:04:28"}
      ],
      "similarity_scores": [[1.0, 0.85], [0.85, 1.0]]
    },
    "B": {
      "segments": [
        {"file": "B1.mp4", "start": "00:15:02", "end": "00:19:55", "duration": "00:04:53"}
      ]
    }
  },
  "ungrouped": []
}
```

---

### Phase 5: Integration & Polish

#### Task 5.1: Full Pipeline Command
- [ ] Create `drumcut process` command that runs entire pipeline
- [ ] Add checkpoint/resume capability (don't redo completed steps)
- [ ] Implement progress reporting with `rich`
- [ ] Add `--dry-run` mode to preview operations

**CLI interface:**
```bash
drumcut process ./session_folder/ \
    --output ./processed/ \
    --filter death-metal \
    --min-song-duration 30 \
    --grouping-threshold auto \
    --keep-intermediates
```

**Pipeline stages:**
1. Discover and validate input files
2. Normalize audio tracks
3. Pan and align audio
4. Mix to stereo master
5. Merge video chapters
6. Sync audio to video
7. Apply video filter
8. Segment by MIDI energy
9. Extract segments
10. Compute similarities
11. Group segments
12. Organize output

#### Task 5.2: Configuration System
- [ ] Create default config file (`~/.config/drumcut/config.yaml`)
- [ ] Allow per-session config override
- [ ] Document all configuration options

**config.yaml:**
```yaml
audio:
  target_lufs: -14
  true_peak: -1
  left_pan: -0.5
  right_pan: 0.5

video:
  chapter_overlap: 0.5
  filter_preset: death-metal

segmentation:
  min_duration: 30
  padding: 3.0
  energy_threshold: auto

grouping:
  method: dtw  # or 'fingerprint'
  threshold: auto
```

#### Task 5.3: Error Handling & Validation
- [ ] Add input validation (file existence, format checks)
- [ ] Implement graceful error recovery
- [ ] Add detailed error messages with suggestions
- [ ] Handle edge cases (empty segments, single segment, etc.)

#### Task 5.4: Testing
- [ ] Create synthetic test fixtures (short audio/video clips)
- [ ] Unit tests for each module
- [ ] Integration test for full pipeline
- [ ] Add CI with GitHub Actions

#### Task 5.5: Documentation
- [ ] Expand README with detailed usage examples
- [ ] Add troubleshooting section
- [ ] Document filter presets with visual examples
- [ ] Add architecture diagram

---

## Future Enhancements (Post-MVP)

- **BPM detection**: Auto-detect tempo of each segment, include in manifest
- **Song identification**: If you have a library of backing tracks, match segments to song names
- **Waveform preview**: Generate waveform images for each segment (useful for quick visual scan)
- **DaVinci integration**: Generate .drp project file or EDL for direct import
- **Web UI**: Simple local web interface for reviewing/adjusting segments before final export
- **Multi-camera support**: Sync multiple GoPro angles
- **Stem separation**: If you record acoustic kit, separate drums from backing track

---

## Technical Notes

### Why These Libraries?

| Library | Purpose | Why Not Alternatives |
|---------|---------|---------------------|
| `librosa` | Audio analysis | Gold standard for MIR, great chroma/DTW |
| `pyloudnorm` | LUFS normalization | Standards-compliant, simple API |
| `soundfile` | Audio I/O | Fast, supports many formats via libsndfile |
| `ffmpeg-python` | Video processing | Thin wrapper, full ffmpeg power |
| `typer` | CLI framework | Clean, composable, well-documented |
| `rich` | Console output | Beautiful progress bars, tables |

### Performance Considerations

- **Memory**: 2-hour session @ 44.1kHz stereo = ~1.4GB in memory. Process in chunks if needed.
- **Disk**: Keep intermediates on fast storage (SSD). Final outputs can go anywhere.
- **CPU**: Chroma extraction is the bottleneck. Consider `n_jobs` parallelization in librosa.
- **GPU**: Not needed for this pipeline. ffmpeg can use hardware encoding but we're mostly copying streams.

### Known Limitations

- Assumes single GoPro (no multi-cam sync)
- Assumes 3-track setup (STL, STR, rendered MIDI) — not generalized to N tracks
- Audio files must be WAV format (Reaper export default)
- Audio filenames must follow `YYYY-MM-DD-<trackname>.wav` pattern with STL/STR suffixes
- Grouping algorithm is greedy, not optimal (but should be good enough)
- No handling of songs that span chapter boundaries (GoPro restart mid-song)

---

## License

MIT — do whatever you want with it.

## Author

Built to solve a very specific problem: spending less time in DaVinci and more time behind the kit. 
