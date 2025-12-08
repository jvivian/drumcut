"""Command-line interface for drumcut."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import IntPrompt
from rich.table import Table

app = typer.Typer(
    name="drumcut",
    help="Automated drum practice video preprocessing.",
    no_args_is_help=True,
)
console = Console()


def _select_session(sessions: dict, session_id: int | None = None) -> tuple[int, list]:
    """Interactive session selection when multiple sessions found.

    Args:
        sessions: Dict mapping session ID to list of GoPro files.
        session_id: Pre-selected session ID (optional).

    Returns:
        Tuple of (selected session_id, list of files).
    """
    if session_id is not None:
        if session_id not in sessions:
            console.print(f"[red]Session {session_id} not found[/]")
            raise typer.Exit(1)
        return session_id, sessions[session_id]

    if len(sessions) == 1:
        session_id = list(sessions.keys())[0]
        return session_id, sessions[session_id]

    # Interactive session selection
    console.print("\n[bold]Multiple sessions found:[/]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Session ID", width=12)
    table.add_column("Chapters", width=10)
    table.add_column("Size", width=10)

    session_ids = sorted(sessions.keys())
    for idx, sid in enumerate(session_ids, 1):
        sfiles = sessions[sid]
        total_size = sum(f.path.stat().st_size for f in sfiles) / 1024 / 1024 / 1024
        table.add_row(str(idx), str(sid), str(len(sfiles)), f"{total_size:.1f} GB")

    console.print(table)
    console.print()

    choice = IntPrompt.ask(
        "[bold]Select session[/]",
        choices=[str(i) for i in range(1, len(session_ids) + 1)],
        console=console,
    )
    session_id = session_ids[choice - 1]
    console.print(f"\n[green]Selected session {session_id}[/]\n")

    return session_id, sessions[session_id]


@app.command()
def process(
    session_folder: Annotated[
        Path, typer.Argument(help="Path to session folder with audio/video files")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "./processed"
    ),
    min_song_duration: Annotated[
        int, typer.Option(help="Minimum song duration in seconds (2+ min default)")
    ] = 120,
    min_misc_duration: Annotated[
        int, typer.Option(help="Minimum misc clip duration in seconds")
    ] = 30,
    buffer: Annotated[float, typer.Option(help="Buffer around songs in seconds")] = 10.0,
    min_silence: Annotated[
        float, typer.Option(help="Minimum silence for song boundary in seconds")
    ] = 3.0,
    session_id: Annotated[
        int | None, typer.Option(help="Specific GoPro session ID to process")
    ] = None,
    skip_mix: Annotated[bool, typer.Option(help="Skip audio mix (use existing)")] = False,
    skip_clips: Annotated[bool, typer.Option(help="Skip clip processing (use existing)")] = False,
    group_similar: Annotated[
        bool, typer.Option(help="Group similar songs by audio fingerprint")
    ] = False,
    keep_intermediates: Annotated[bool, typer.Option(help="Keep intermediate files")] = False,
    dry_run: Annotated[bool, typer.Option(help="Preview operations without executing")] = False,
) -> None:
    """Run the full preprocessing pipeline on a session folder.

    New memory-efficient pipeline:
    1. Mix audio tracks (L, R, MIDI) to stereo
    2. Detect song boundaries (silence across ALL tracks)
    3. Process each clip individually (align + replace audio)
    4. Export songs (2+ min) and misc clips (30s-2min)
    """
    import shutil

    import librosa

    from drumcut.audio.io import detect_track_roles
    from drumcut.audio.mix import mix_session_ffmpeg
    from drumcut.segmentation.detect import detect_song_boundaries, segments_from_boundaries
    from drumcut.ui import PipelineUI
    from drumcut.video.gopro import find_gopro_files, group_by_session

    # Create output directories
    output = Path(output)
    intermediates_dir = output / "intermediates"
    clips_dir = intermediates_dir / "clips"

    # Pre-flight: Find GoPro files and validate
    gopro_files = find_gopro_files(session_folder)
    if not gopro_files:
        console.print("[red]No GoPro files found[/]")
        raise typer.Exit(1)

    sessions = group_by_session(gopro_files)

    # Select session (interactive if multiple)
    session_id, selected_files = _select_session(sessions, session_id)

    # Find audio files
    audio_files = list(session_folder.glob("*.wav")) + list(session_folder.glob("*.WAV"))
    roles = detect_track_roles(audio_files) if audio_files else {}

    if dry_run:
        # Simple dry run output
        console.print("\n[bold]🥁 drumcut Pipeline[/] [dim](dry run)[/]\n")
        console.print(f"  Session: {session_folder}")
        console.print(f"  Output:  {output}")
        console.print(f"  GoPro:   Session {session_id} ({len(selected_files)} chapters)")
        console.print(f"  Audio:   {list(roles.keys()) if roles else 'None found'}")
        console.print()
        console.print("  [dim]1.[/] Mix Audio      → Mix tracks to stereo")
        console.print("  [dim]2.[/] Song Boundaries → Detect silence across all tracks")
        console.print("  [dim]3.[/] Process Clips  → Align + replace audio per clip")
        console.print("  [dim]4.[/] Export Songs   → songs/ (2+ min) and misc/ (30s-2min)")
        if group_similar:
            console.print("  [dim]5.[/] Group Similar  → Cluster by audio fingerprint")
        console.print()
        return

    if not roles:
        console.print("[red]No audio files found[/]")
        raise typer.Exit(1)

    # Create directories
    output.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(exist_ok=True)
    clips_dir.mkdir(exist_ok=True)

    mixed_audio = intermediates_dir / "mixed.wav"

    # Create UI
    ui = PipelineUI(title="🥁 drumcut Pipeline")
    ui.add_step("Mix Audio", f"Mix {len(roles)} audio tracks")
    ui.add_step("Song Boundaries", "Detect silence across all tracks")
    ui.add_step("Process Clips", f"Process {len(selected_files)} clips")
    ui.add_step("Export Songs", "Export songs/ and misc/")
    if group_similar:
        ui.add_step("Group Similar", "Cluster by audio fingerprint")

    with ui:
        # Step 1: Mix audio
        if skip_mix and mixed_audio.exists():
            ui.skip_step(0, "Using existing")
        else:
            ui.start_step(0, f"Mixing {len(roles)} tracks...")
            ui.add_substep("Aligning tracks...")
            mix_session_ffmpeg(session_folder, mixed_audio, verbose=False)
            ui.add_substep("Normalizing to -14 LUFS...")
            ui.complete_step(0, f"{', '.join(roles.keys())}")

        # Step 2: Detect song boundaries using ALL tracks
        ui.start_step(1, "Analyzing tracks for silence...")
        track_paths = list(roles.values())

        # Show progress per track
        ui.start_progress("Analyzing tracks", total=len(track_paths))
        for role in roles:
            ui.add_substep(f"Analyzing {role} track...")
            ui.advance_progress()

        boundaries = detect_song_boundaries(
            track_paths,
            min_silence_duration=min_silence,
        )
        ui.complete_progress()

        # Get total duration from mixed audio
        mixed_duration = librosa.get_duration(path=str(mixed_audio))

        # Convert boundaries to segments
        segments = segments_from_boundaries(
            boundaries,
            total_duration=mixed_duration,
            min_duration=min_misc_duration,
            buffer_seconds=buffer,
        )
        ui.complete_step(1, f"{len(boundaries)} boundaries → {len(segments)} segments")

        # Step 3: Process each clip individually
        from drumcut.video.clip import ClipInfo, process_clip

        video_paths = [f.path for f in selected_files]
        if skip_clips and clips_dir.exists() and any(clips_dir.iterdir()):
            ui.skip_step(2, "Using existing")
            processed_files = sorted(clips_dir.glob("processed_*.mp4"))
            clip_infos = []
            cumulative_offset = 0.0
            for pf in processed_files:
                duration = librosa.get_duration(path=str(pf))
                clip_infos.append(
                    ClipInfo(
                        path=pf,
                        offset_seconds=cumulative_offset,
                        duration_seconds=duration,
                        correlation_strength=1.0,
                    )
                )
                cumulative_offset += duration
        else:
            ui.start_step(2, f"Processing {len(video_paths)} clips...")
            ui.start_progress("Processing clips", total=len(video_paths))

            clip_infos = []
            for i, video_path in enumerate(video_paths, 1):
                ui.add_substep(f"Clip {i}/{len(video_paths)}: {video_path.name}")
                output_path = clips_dir / f"processed_{video_path.name}"

                clip_info = process_clip(
                    video_path=video_path,
                    mixed_audio_path=mixed_audio,
                    output_path=output_path,
                    alignment_window=30.0,
                    verbose=False,
                )
                clip_infos.append(clip_info)
                ui.advance_progress()

            ui.complete_progress()
            ui.complete_step(2, f"{len(clip_infos)} clips processed")

        # Step 4: Export songs
        from drumcut.export import (
            SongExport,
            categorize_segment,
            concat_clips,
            extract_segment_from_clip,
            map_segment_to_clips,
        )

        ui.start_step(3, "Exporting songs...")

        songs_dir = output / "songs"
        misc_dir = output / "misc"
        songs_dir.mkdir(parents=True, exist_ok=True)
        misc_dir.mkdir(parents=True, exist_ok=True)

        # Filter valid segments
        valid_segments = [s for s in segments if s.duration_seconds >= min_misc_duration]
        ui.start_progress("Exporting", total=len(valid_segments))

        exports = []
        song_count = 0
        misc_count = 0

        for segment in valid_segments:
            clip_mappings = map_segment_to_clips(segment, clip_infos)
            if not clip_mappings:
                ui.advance_progress()
                continue

            category = categorize_segment(segment, float(min_song_duration))
            spans_clips = len(clip_mappings) > 1

            if category == "songs":
                song_count += 1
                output_path = songs_dir / f"song_{song_count:03d}.mp4"
                ui.add_substep(f"Song {song_count}: {segment.duration_seconds / 60:.1f} min")
            else:
                misc_count += 1
                output_path = misc_dir / f"misc_{misc_count:03d}.mp4"
                ui.add_substep(f"Misc {misc_count}: {segment.duration_seconds:.0f}s")

            if spans_clips:
                temp_clips = []
                temp_dir = output / "temp"
                temp_dir.mkdir(exist_ok=True)
                for j, (clip, start, end) in enumerate(clip_mappings):
                    temp_path = temp_dir / f"part_{j:03d}.mp4"
                    extract_segment_from_clip(clip.path, temp_path, start, end)
                    temp_clips.append(temp_path)
                concat_clips(temp_clips, output_path)
                for tp in temp_clips:
                    tp.unlink(missing_ok=True)
                temp_dir.rmdir()
            else:
                clip, start, end = clip_mappings[0]
                extract_segment_from_clip(clip.path, output_path, start, end)

            exports.append(
                SongExport(
                    path=output_path,
                    duration_seconds=segment.duration_seconds,
                    category=category,
                    source_clips=[m[0].path.name for m in clip_mappings],
                    spans_clips=spans_clips,
                )
            )
            ui.advance_progress()

        ui.complete_progress()
        songs = [e for e in exports if e.category == "songs"]
        misc = [e for e in exports if e.category == "misc"]
        ui.complete_step(3, f"{len(songs)} songs, {len(misc)} misc")

        # Step 5: Group similar songs (optional)
        groups = {}
        if group_similar and len(songs) >= 2:
            import subprocess
            import tempfile

            from drumcut.grouping.cluster import cluster_segments
            from drumcut.grouping.similarity import compute_distance_matrix

            ui.start_step(4, "Extracting audio fingerprints...")

            # Extract audio from each song for fingerprinting
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                audio_paths = []

                ui.start_progress("Extracting audio", total=len(songs))
                for song in songs:
                    audio_path = tmpdir / f"{song.path.stem}.wav"
                    ui.add_substep(f"Extracting {song.path.name}...")
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(song.path),
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "22050",
                            str(audio_path),
                        ],
                        capture_output=True,
                    )
                    audio_paths.append(audio_path)
                    ui.advance_progress()
                ui.complete_progress()

                # Compute similarity matrix
                ui.add_substep("Computing audio similarity...")
                ui.start_progress("Computing DTW distances", total=None)
                distances = compute_distance_matrix(audio_paths)
                ui.complete_progress()

                # Cluster
                ui.add_substep("Clustering similar songs...")
                groups = cluster_segments(distances)

            # Organize into group folders
            ui.add_substep(f"Organizing into {len(groups)} groups...")
            for label, indices in groups.items():
                group_dir = songs_dir / label
                group_dir.mkdir(exist_ok=True)
                for idx in indices:
                    song = songs[idx]
                    new_path = group_dir / song.path.name
                    song.path.rename(new_path)
                    song.path = new_path

            ui.complete_step(4, f"{len(groups)} groups")
        elif group_similar:
            ui.skip_step(4, "Need ≥2 songs")

    # Cleanup intermediates
    if not keep_intermediates and intermediates_dir.exists():
        shutil.rmtree(intermediates_dir)

    # Summary
    console.print()
    console.print("[bold green]✓ Pipeline complete![/]")
    console.print(f"  Output: {output}")
    console.print()

    # Songs summary
    if songs:
        if groups:
            console.print(f"  [bold]Songs[/] ({len(songs)} in {len(groups)} groups):")
            for label, indices in sorted(groups.items()):
                console.print(f"    [bold cyan]Group {label}[/] ({len(indices)} songs):")
                for idx in indices:
                    s = songs[idx]
                    duration_min = s.duration_seconds / 60
                    if s.spans_clips:
                        clips_str = f"[yellow]spans {len(s.source_clips)} clips[/]"
                    else:
                        clips_str = f"[dim]{s.source_clips[0]}[/]"
                    console.print(f"      {s.path.name}: {duration_min:.1f} min ({clips_str})")
        else:
            console.print(f"  [bold]Songs[/] ({len(songs)} in songs/):")
            for s in songs:
                duration_min = s.duration_seconds / 60
                if s.spans_clips:
                    clips_str = f"[yellow]spans {len(s.source_clips)} clips[/]"
                else:
                    clips_str = f"[dim]{s.source_clips[0]}[/]"
                console.print(f"    {s.path.name}: {duration_min:.1f} min ({clips_str})")

    # Misc summary
    if misc:
        console.print(f"  [bold]Misc[/] ({len(misc)} in misc/):")
        for m in misc:
            console.print(f"    {m.path.name}: {m.duration_seconds:.0f}s")


@app.command()
def normalize(
    audio_file: Annotated[Path, typer.Argument(help="Audio file to normalize")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")] = None,
    target_lufs: Annotated[float, typer.Option(help="Target loudness in LUFS")] = -14.0,
    true_peak: Annotated[float, typer.Option(help="True peak ceiling in dBTP")] = -1.0,
) -> None:
    """Normalize audio to target LUFS with true peak limiting."""
    from drumcut.audio.normalize import normalize_audio

    output_path = output or audio_file.with_stem(f"{audio_file.stem}_normalized")
    console.print(f"[bold blue]Normalizing:[/] {audio_file}")
    console.print(f"[bold blue]Target:[/] {target_lufs} LUFS, {true_peak} dBTP")

    result = normalize_audio(audio_file, output_path, target_lufs, true_peak)
    console.print(f"[green]Saved to:[/] {output_path}")
    console.print(f"  Input LUFS: {result['input_lufs']:.1f}")
    console.print(f"  Output LUFS: {result['output_lufs']:.1f}")
    console.print(f"  Gain applied: {result['gain_applied_db']:.1f} dB")
    console.print(f"  True peak: {result['true_peak_dbtp']:.1f} dBTP")


@app.command()
def mix(
    session_folder: Annotated[Path, typer.Argument(help="Session folder with audio files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path(
        "./mixed.wav"
    ),
    left_pan: Annotated[float, typer.Option(help="Left track pan position (-1 to 1)")] = -0.5,
    right_pan: Annotated[float, typer.Option(help="Right track pan position (-1 to 1)")] = 0.5,
    target_lufs: Annotated[float, typer.Option(help="Target loudness in LUFS")] = -14.0,
) -> None:
    """Mix and pan audio tracks from a session folder."""
    from drumcut.audio.mix import mix_session

    console.print(f"[bold blue]Mixing session:[/] {session_folder}")
    console.print(f"[bold blue]Pan settings:[/] L={left_pan}, R={right_pan}")
    console.print(f"[bold blue]Target:[/] {target_lufs} LUFS")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Mixing audio tracks...", total=None)
        result = mix_session(
            session_folder,
            output,
            left_pan=left_pan,
            right_pan=right_pan,
            target_lufs=target_lufs,
            verbose=False,
        )

    console.print(f"[green]Saved to:[/] {output}")
    console.print(f"  Tracks mixed: {', '.join(result['tracks'])}")
    console.print(f"  Output LUFS: {result['normalization']['output_lufs']:.1f}")
    console.print(f"  True peak: {result['normalization']['true_peak_dbtp']:.1f} dBTP")


@app.command("merge-video")
def merge_video(
    session_folder: Annotated[Path, typer.Argument(help="Session folder with GoPro files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path(
        "./merged.mp4"
    ),
    overlap: Annotated[
        str, typer.Option(help="Overlap handling: 'auto' or seconds to trim")
    ] = "auto",
    session_id: Annotated[
        int | None, typer.Option(help="Specific session ID to merge (optional)")
    ] = None,
) -> None:
    """Merge GoPro video chapters into a single file."""
    from drumcut.video.gopro import find_gopro_files, group_by_session
    from drumcut.video.merge import merge_chapters

    console.print(f"[bold blue]Scanning for GoPro files:[/] {session_folder}")

    files = find_gopro_files(session_folder)
    if not files:
        console.print("[red]No GoPro files found[/]")
        raise typer.Exit(1)

    sessions = group_by_session(files)

    # Select session (interactive if multiple)
    session_id, selected_files = _select_session(sessions, session_id)

    console.print(f"[bold blue]Merging session {session_id}:[/] {len(selected_files)} chapters")

    # Parse overlap setting
    overlap_trim = overlap if overlap == "auto" else float(overlap)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Merging video chapters...", total=None)
        result = merge_chapters(selected_files, output, overlap_trim=overlap_trim, verbose=True)

    output_size = result.stat().st_size / 1024 / 1024 / 1024
    console.print(f"[green]Saved to:[/] {result}")
    console.print(f"  Size: {output_size:.2f} GB")


@app.command()
def sync(
    video_file: Annotated[Path, typer.Argument(help="Video file")],
    audio_file: Annotated[Path, typer.Argument(help="Audio file to sync")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path(
        "./synced.mp4"
    ),
) -> None:
    """Sync external audio to video using cross-correlation."""
    from drumcut.video.sync import sync_audio_to_video

    console.print(f"[bold blue]Video:[/] {video_file}")
    console.print(f"[bold blue]Audio:[/] {audio_file}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Syncing audio to video...", total=None)
        result = sync_audio_to_video(video_file, audio_file, output)

    console.print(f"[green]Saved to:[/] {output}")
    console.print(f"  Offset: {result['offset_seconds']:.3f}s")
    console.print(f"  Correlation: {result['correlation_strength']:.3f}")


@app.command()
def segment(
    video_file: Annotated[Path, typer.Argument(help="Video file to segment")],
    midi_track: Annotated[
        Path | None, typer.Option(help="MIDI/drums track for energy detection")
    ] = None,
    audio_dir: Annotated[
        Path | None, typer.Option(help="Directory to auto-detect MIDI track")
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "./segments"
    ),
    min_duration: Annotated[int, typer.Option(help="Minimum segment duration in seconds")] = 30,
    padding: Annotated[float, typer.Option(help="Padding around segments in seconds")] = 3.0,
    dry_run: Annotated[
        bool, typer.Option(help="Show detected segments without extracting")
    ] = False,
) -> None:
    """Segment video based on MIDI track energy."""
    from drumcut.audio.io import detect_track_roles, load_audio
    from drumcut.segmentation.detect import detect_activity_regions
    from drumcut.segmentation.slice import extract_segments

    console.print(f"[bold blue]Segmenting:[/] {video_file}")

    # Find MIDI track
    if midi_track is None:
        if audio_dir is None:
            console.print("[red]Must provide --midi-track or --audio-dir[/]")
            raise typer.Exit(1)

        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.WAV"))
        roles = detect_track_roles(audio_files)

        if "midi" not in roles:
            console.print("[red]No MIDI track found in audio directory[/]")
            raise typer.Exit(1)

        midi_track = roles["midi"]

    console.print(f"[bold blue]MIDI track:[/] {midi_track}")

    # Load and analyze
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing audio energy...", total=None)
        audio, sr = load_audio(midi_track)
        segments = detect_activity_regions(
            audio,
            sr,
            min_duration=min_duration,
            padding=padding,
        )

    if not segments:
        console.print("[yellow]No segments detected[/]")
        raise typer.Exit(0)

    console.print(f"\n[green]Found {len(segments)} segment(s):[/]")
    for i, seg in enumerate(segments, 1):
        mins, secs = divmod(seg.start_seconds, 60)
        end_mins, end_secs = divmod(seg.end_seconds, 60)
        console.print(
            f"  {i}. [{int(mins):02d}:{secs:05.2f} → {int(end_mins):02d}:{end_secs:05.2f}] "
            f"({seg.duration_seconds:.1f}s)"
        )

    if dry_run:
        console.print("\n[yellow]Dry run - no segments extracted[/]")
        return

    # Extract segments
    console.print(f"\n[bold blue]Extracting to:[/] {output}")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Extracting segments...", total=None)
        output_paths = extract_segments(video_file, segments, output)

    console.print(f"[green]Extracted {len(output_paths)} segment(s)[/]")


@app.command()
def group(
    segments_dir: Annotated[Path, typer.Argument(help="Directory containing segments")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "./grouped"
    ),
    threshold: Annotated[
        float | None, typer.Option(help="Similarity threshold (auto if not set)")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Show groupings without copying files")] = False,
) -> None:
    """Group similar segments by audio fingerprinting."""
    import subprocess
    import tempfile

    from drumcut.grouping.cluster import cluster_segments, organize_output
    from drumcut.grouping.similarity import compute_distance_matrix

    console.print(f"[bold blue]Grouping segments from:[/] {segments_dir}")

    # Find video segments
    segment_paths = sorted(list(segments_dir.glob("*.mp4")) + list(segments_dir.glob("*.MP4")))

    if not segment_paths:
        console.print("[red]No video segments found[/]")
        raise typer.Exit(1)

    console.print(f"Found {len(segment_paths)} segment(s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract audio from each segment
        console.print("[bold blue]Extracting audio for fingerprinting...[/]")
        audio_paths = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting audio...", total=len(segment_paths))

            for seg_path in segment_paths:
                audio_path = tmpdir / f"{seg_path.stem}.wav"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(seg_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "22050",
                    str(audio_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    console.print(
                        f"[yellow]Warning: Failed to extract audio from {seg_path.name}[/]"
                    )
                    continue
                audio_paths.append(audio_path)
                progress.advance(task)

        if len(audio_paths) < 2:
            console.print("[yellow]Need at least 2 segments for grouping[/]")
            raise typer.Exit(0)

        # Compute similarity matrix
        console.print("[bold blue]Computing audio similarity...[/]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Computing DTW distances...", total=None)
            distances = compute_distance_matrix(audio_paths)

        # Cluster
        groups = cluster_segments(distances, threshold=threshold)

        # Display results
        console.print(f"\n[green]Found {len(groups)} group(s):[/]")
        for label, indices in groups.items():
            names = [segment_paths[i].name for i in indices]
            console.print(f"  Group {label}: {', '.join(names)}")

        if dry_run:
            console.print("\n[yellow]Dry run - files not copied[/]")
            return

        # Organize output
        console.print(f"\n[bold blue]Organizing output to:[/] {output}")
        organize_output(segment_paths, groups, output)

        console.print(f"[green]Created {len(groups)} group folder(s)[/]")


@app.command()
def version() -> None:
    """Show version information."""
    from drumcut import __version__

    console.print(f"drumcut version {__version__}")


if __name__ == "__main__":
    app()
