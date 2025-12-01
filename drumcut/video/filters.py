"""Video filter presets."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Filter preset definitions
FILTER_PRESETS: dict[str, str] = {
    "death-metal": "hue=s=0,eq=contrast=1.2:brightness=-0.03,noise=c0s=8:c0f=t",
    "clean": "hue=s=0",
    "raw": "null",
    "film": "hue=s=0,curves=m='0/0 0.1/0.05 0.9/0.95 1/1',vignette=PI/5",
    "high-contrast": "hue=s=0,eq=contrast=1.4:brightness=-0.05",
    "warm": "colorbalance=rs=0.1:gs=0.05:bs=-0.1,eq=saturation=0.8",
}


def get_filter_string(preset: str | None = None, custom: str | None = None) -> str:
    """
    Get ffmpeg filter string from preset name or custom string.

    Args:
        preset: Name of preset from FILTER_PRESETS.
        custom: Custom ffmpeg filter string (overrides preset).

    Returns:
        Filter string for ffmpeg -vf option.

    Raises:
        ValueError: If neither preset nor custom is provided.
    """
    if custom:
        return custom

    if preset:
        if preset not in FILTER_PRESETS:
            available = ", ".join(FILTER_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
        return FILTER_PRESETS[preset]

    raise ValueError("Must provide either preset or custom filter")


def apply_filter(
    input_path: Path | str,
    output_path: Path | str,
    preset: str | None = None,
    custom: str | None = None,
    crf: int = 18,
) -> None:
    """
    Apply video filter to input and save to output.

    Args:
        input_path: Input video path.
        output_path: Output video path.
        preset: Filter preset name.
        custom: Custom ffmpeg filter string.
        crf: Constant Rate Factor for encoding quality (lower = better, 18 is visually lossless).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_string = get_filter_string(preset, custom)

    # If raw/null filter, just copy
    if filter_string == "null":
        import shutil

        shutil.copy(input_path, output_path)
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_string,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-c:a",
        "copy",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg filter failed: {result.stderr}")


def preview_filter(
    input_path: Path | str,
    preset: str | None = None,
    custom: str | None = None,
    duration: float = 5.0,
) -> None:
    """
    Preview filter on a short clip using ffplay.

    Args:
        input_path: Input video path.
        preset: Filter preset name.
        custom: Custom ffmpeg filter string.
        duration: Seconds to preview.
    """
    input_path = Path(input_path)
    filter_string = get_filter_string(preset, custom)

    cmd = [
        "ffplay",
        "-t",
        str(duration),
        "-vf",
        filter_string,
        str(input_path),
    ]

    subprocess.run(cmd)
