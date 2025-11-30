"""Command-line interface for drumcut."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="drumcut",
    help="Automated drum practice video preprocessing.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def process(
    session_folder: Annotated[Path, typer.Argument(help="Path to session folder with audio/video files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("./processed"),
    filter_preset: Annotated[str, typer.Option("--filter", "-f", help="Video filter preset")] = "death-metal",
    min_duration: Annotated[int, typer.Option(help="Minimum song duration in seconds")] = 30,
    padding: Annotated[float, typer.Option(help="Padding around segments in seconds")] = 3.0,
    keep_intermediates: Annotated[bool, typer.Option(help="Keep intermediate files")] = False,
    dry_run: Annotated[bool, typer.Option(help="Preview operations without executing")] = False,
) -> None:
    """Run the full preprocessing pipeline on a session folder."""
    console.print(f"[bold blue]Processing session:[/] {session_folder}")
    console.print(f"[bold blue]Output directory:[/] {output}")
    console.print(f"[bold blue]Filter preset:[/] {filter_preset}")

    if dry_run:
        console.print("[yellow]Dry run mode - no operations will be performed[/]")
        return

    # TODO: Implement full pipeline
    console.print("[red]Full pipeline not yet implemented[/]")


@app.command()
def normalize(
    audio_file: Annotated[Path, typer.Argument(help="Audio file to normalize")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path")] = None,
    target_lufs: Annotated[float, typer.Option(help="Target loudness in LUFS")] = -14.0,
    true_peak: Annotated[float, typer.Option(help="True peak ceiling in dBTP")] = -1.0,
) -> None:
    """Normalize audio to target LUFS with true peak limiting."""
    from drumcut.audio.normalize import normalize_audio

    output_path = output or audio_file.with_stem(f"{audio_file.stem}_normalized")
    console.print(f"[bold blue]Normalizing:[/] {audio_file}")
    console.print(f"[bold blue]Target:[/] {target_lufs} LUFS, {true_peak} dBTP")

    normalize_audio(audio_file, output_path, target_lufs, true_peak)
    console.print(f"[green]Saved to:[/] {output_path}")


@app.command()
def mix(
    session_folder: Annotated[Path, typer.Argument(help="Session folder with audio files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("./mixed.wav"),
    left_pan: Annotated[float, typer.Option(help="Left track pan position (-1 to 1)")] = -0.5,
    right_pan: Annotated[float, typer.Option(help="Right track pan position (-1 to 1)")] = 0.5,
) -> None:
    """Mix and pan audio tracks from a session folder."""
    console.print(f"[bold blue]Mixing session:[/] {session_folder}")
    console.print(f"[bold blue]Pan settings:[/] L={left_pan}, R={right_pan}")

    # TODO: Implement mixing
    console.print("[red]Mixing not yet implemented[/]")


@app.command("merge-video")
def merge_video(
    session_folder: Annotated[Path, typer.Argument(help="Session folder with GoPro files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("./merged.mp4"),
    overlap_trim: Annotated[float, typer.Option(help="Seconds to trim from chapter overlaps")] = 0.5,
) -> None:
    """Merge GoPro video chapters into a single file."""
    console.print(f"[bold blue]Merging videos from:[/] {session_folder}")

    # TODO: Implement video merging
    console.print("[red]Video merging not yet implemented[/]")


@app.command()
def sync(
    video_file: Annotated[Path, typer.Argument(help="Video file")],
    audio_file: Annotated[Path, typer.Argument(help="Audio file to sync")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("./synced.mp4"),
) -> None:
    """Sync external audio to video using cross-correlation."""
    console.print(f"[bold blue]Syncing:[/] {audio_file} to {video_file}")

    # TODO: Implement sync
    console.print("[red]Sync not yet implemented[/]")


@app.command()
def segment(
    video_file: Annotated[Path, typer.Argument(help="Video file to segment")],
    midi_track: Annotated[Optional[Path], typer.Option(help="MIDI/drums track for energy detection")] = None,
    audio_dir: Annotated[Optional[Path], typer.Option(help="Directory to auto-detect MIDI track")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("./segments"),
    min_duration: Annotated[int, typer.Option(help="Minimum segment duration in seconds")] = 30,
    padding: Annotated[float, typer.Option(help="Padding around segments in seconds")] = 3.0,
) -> None:
    """Segment video based on MIDI track energy."""
    console.print(f"[bold blue]Segmenting:[/] {video_file}")

    # TODO: Implement segmentation
    console.print("[red]Segmentation not yet implemented[/]")


@app.command()
def group(
    segments_dir: Annotated[Path, typer.Argument(help="Directory containing segments")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("./grouped"),
    threshold: Annotated[Optional[float], typer.Option(help="Similarity threshold (auto if not set)")] = None,
) -> None:
    """Group similar segments by audio fingerprinting."""
    console.print(f"[bold blue]Grouping segments from:[/] {segments_dir}")

    # TODO: Implement grouping
    console.print("[red]Grouping not yet implemented[/]")


@app.command()
def version() -> None:
    """Show version information."""
    from drumcut import __version__
    console.print(f"drumcut version {__version__}")


if __name__ == "__main__":
    app()
