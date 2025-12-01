"""Command-line interface for drumcut."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="drumcut",
    help="Automated drum practice video preprocessing.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def process(
    session_folder: Annotated[
        Path, typer.Argument(help="Path to session folder with audio/video files")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "./processed"
    ),
    filter_preset: Annotated[
        str, typer.Option("--filter", "-f", help="Video filter preset")
    ] = "death-metal",
    min_duration: Annotated[int, typer.Option(help="Minimum song duration in seconds")] = 30,
    padding: Annotated[float, typer.Option(help="Padding around segments in seconds")] = 3.0,
    session_id: Annotated[
        int | None, typer.Option(help="Specific GoPro session ID to process")
    ] = None,
    skip_merge: Annotated[bool, typer.Option(help="Skip video merge (use existing)")] = False,
    skip_mix: Annotated[bool, typer.Option(help="Skip audio mix (use existing)")] = False,
    skip_sync: Annotated[bool, typer.Option(help="Skip audio-video sync")] = False,
    skip_segment: Annotated[bool, typer.Option(help="Skip segmentation")] = False,
    skip_group: Annotated[bool, typer.Option(help="Skip grouping")] = False,
    keep_intermediates: Annotated[bool, typer.Option(help="Keep intermediate files")] = False,
    dry_run: Annotated[bool, typer.Option(help="Preview operations without executing")] = False,
) -> None:
    """Run the full preprocessing pipeline on a session folder.

    Pipeline steps:
    1. Merge GoPro chapters into single video
    2. Mix audio tracks (L, R, MIDI) to stereo
    3. Sync mixed audio to video
    4. Segment video by song boundaries
    5. Group similar segments (optional)
    """
    import shutil
    import subprocess
    import tempfile

    from drumcut.audio.io import detect_track_roles, load_audio
    from drumcut.audio.mix import mix_session
    from drumcut.grouping.cluster import cluster_segments, organize_output
    from drumcut.grouping.similarity import compute_distance_matrix
    from drumcut.segmentation.detect import detect_activity_regions
    from drumcut.segmentation.slice import extract_segments
    from drumcut.ui import PipelineUI
    from drumcut.video.gopro import find_gopro_files, group_by_session
    from drumcut.video.merge import merge_chapters
    from drumcut.video.sync import sync_audio_to_video

    # Create output directories
    output = Path(output)
    intermediates_dir = output / "intermediates"
    segments_dir = output / "segments"
    grouped_dir = output / "grouped"

    # Pre-flight: Find GoPro files and validate
    gopro_files = find_gopro_files(session_folder)
    if not gopro_files:
        console.print("[red]No GoPro files found[/]")
        raise typer.Exit(1)

    sessions = group_by_session(gopro_files)

    # Select session
    if session_id is not None:
        if session_id not in sessions:
            console.print(f"[red]Session {session_id} not found[/]")
            raise typer.Exit(1)
        selected_files = sessions[session_id]
    elif len(sessions) == 1:
        session_id = list(sessions.keys())[0]
        selected_files = sessions[session_id]
    else:
        console.print("[yellow]Multiple sessions found. Use --session-id to select one.[/]")
        for sid, sfiles in sessions.items():
            total_size = sum(f.path.stat().st_size for f in sfiles) / 1024 / 1024 / 1024
            console.print(f"  Session {sid}: {len(sfiles)} chapters ({total_size:.1f} GB)")
        raise typer.Exit(1)

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
        console.print("  [dim]1.[/] Merge Video    → Would merge chapters")
        console.print("  [dim]2.[/] Mix Audio      → Would mix tracks")
        console.print("  [dim]3.[/] Sync A/V       → Would sync audio")
        console.print("  [dim]4.[/] Segment        → Would detect songs")
        console.print("  [dim]5.[/] Group          → Would cluster similar")
        console.print()
        return

    # Create directories
    output.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(exist_ok=True)

    merged_video = intermediates_dir / "merged.mp4"
    mixed_audio = intermediates_dir / "mixed.wav"
    synced_video = intermediates_dir / "synced.mp4"

    # Create UI
    ui = PipelineUI(title="🥁 drumcut Pipeline")
    ui.add_step("Merge Video", f"Merge {len(selected_files)} GoPro chapters")
    ui.add_step("Mix Audio", f"Mix {len(roles)} audio tracks" if roles else "No audio")
    ui.add_step("Sync A/V", "Sync audio to video")
    ui.add_step("Segment", "Detect song boundaries")
    ui.add_step("Group", "Cluster similar segments")

    segments = []
    segment_paths = []

    with ui:
        # Step 1: Merge video
        if skip_merge and merged_video.exists():
            ui.skip_step(0, "Using existing")
        else:
            ui.start_step(0, f"Merging {len(selected_files)} chapters...")
            merge_chapters(selected_files, merged_video, overlap_trim="auto", verbose=False)
            size_gb = merged_video.stat().st_size / 1024 / 1024 / 1024
            ui.complete_step(0, f"{size_gb:.1f} GB")

        # Step 2: Mix audio
        if not roles:
            ui.skip_step(1, "No audio files")
            skip_mix = True
            skip_sync = True
        elif skip_mix and mixed_audio.exists():
            ui.skip_step(1, "Using existing")
        else:
            ui.start_step(1, f"Mixing {len(roles)} tracks...")
            mix_session(session_folder, mixed_audio, verbose=False)
            ui.complete_step(1, f"{', '.join(roles.keys())}")

        # Step 3: Sync audio to video
        if skip_sync or skip_mix:
            ui.skip_step(2, "Skipped")
            synced_video = merged_video
        else:
            ui.start_step(2, "Finding sync offset...")
            result = sync_audio_to_video(merged_video, mixed_audio, synced_video)
            offset_ms = result["offset_seconds"] * 1000
            ui.complete_step(2, f"Offset: {offset_ms:+.0f}ms")

        # Step 4: Segment
        midi_track = roles.get("midi") if roles else None
        if skip_segment:
            ui.skip_step(3, "Skipped")
        elif midi_track is None:
            ui.skip_step(3, "No MIDI track")
        else:
            ui.start_step(3, "Analyzing energy...")
            audio, sr = load_audio(midi_track)
            segments = detect_activity_regions(
                audio, sr, min_duration=min_duration, padding=padding
            )
            ui.update_step(f"Extracting {len(segments)} segments...")

            if segments:
                segment_paths = extract_segments(synced_video, segments, segments_dir)
                ui.complete_step(3, f"{len(segments)} segments")
            else:
                ui.complete_step(3, "No segments found")

        # Step 5: Group
        if skip_group or not segment_paths:
            ui.skip_step(4, "Skipped")
        elif len(segment_paths) < 2:
            ui.skip_step(4, "Need ≥2 segments")
        else:
            ui.start_step(4, "Extracting audio...")
            segment_files = sorted(segments_dir.glob("*.mp4"))

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                audio_paths = []

                for seg_path in segment_files:
                    audio_path = tmpdir / f"{seg_path.stem}.wav"
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(seg_path),
                            "-vn", "-ac", "1", "-ar", "22050", str(audio_path),
                        ],
                        capture_output=True,
                    )
                    audio_paths.append(audio_path)

                ui.update_step("Computing similarity...")
                distances = compute_distance_matrix(audio_paths)
                groups = cluster_segments(distances)

                ui.update_step("Organizing output...")
                organize_output(segment_files, groups, grouped_dir)
                ui.complete_step(4, f"{len(groups)} groups")

    # Cleanup intermediates
    if not keep_intermediates and intermediates_dir.exists():
        shutil.rmtree(intermediates_dir)

    # Summary
    console.print()
    console.print("[bold green]✓ Pipeline complete![/]")
    console.print(f"  Output: {output}")
    if segments:
        console.print(f"  Segments: {len(segments)}")
    if segment_paths and not skip_group:
        console.print(f"  Groups: {grouped_dir}")


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
    console.print(f"Found {len(sessions)} session(s): {list(sessions.keys())}")

    # Select session
    if session_id is not None:
        if session_id not in sessions:
            console.print(f"[red]Session {session_id} not found[/]")
            raise typer.Exit(1)
        selected_files = sessions[session_id]
    elif len(sessions) == 1:
        session_id = list(sessions.keys())[0]
        selected_files = sessions[session_id]
    else:
        console.print("[yellow]Multiple sessions found. Use --session-id to select one.[/]")
        for sid, sfiles in sessions.items():
            total_size = sum(f.path.stat().st_size for f in sfiles) / 1024 / 1024 / 1024
            console.print(f"  Session {sid}: {len(sfiles)} chapters ({total_size:.1f} GB)")
        raise typer.Exit(1)

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
