"""Configuration management for drumcut."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    target_lufs: float = -14.0
    true_peak: float = -1.0
    left_pan: float = -0.5
    right_pan: float = 0.5
    sample_rate: int = 44100


@dataclass
class VideoConfig:
    """Video processing configuration."""
    chapter_overlap: float = 0.5
    filter_preset: str = "death-metal"


@dataclass
class SegmentationConfig:
    """Segmentation configuration."""
    min_duration: int = 30
    padding: float = 3.0
    energy_threshold: Optional[float] = None  # None = auto


@dataclass
class GroupingConfig:
    """Grouping configuration."""
    method: str = "dtw"
    threshold: Optional[float] = None  # None = auto


@dataclass
class Config:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file."""
        if path is None:
            path = Path.home() / ".config" / "drumcut" / "config.yaml"

        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            video=VideoConfig(**data.get("video", {})),
            segmentation=SegmentationConfig(**data.get("segmentation", {})),
            grouping=GroupingConfig(**data.get("grouping", {})),
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if path is None:
            path = Path.home() / ".config" / "drumcut" / "config.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "audio": {
                "target_lufs": self.audio.target_lufs,
                "true_peak": self.audio.true_peak,
                "left_pan": self.audio.left_pan,
                "right_pan": self.audio.right_pan,
                "sample_rate": self.audio.sample_rate,
            },
            "video": {
                "chapter_overlap": self.video.chapter_overlap,
                "filter_preset": self.video.filter_preset,
            },
            "segmentation": {
                "min_duration": self.segmentation.min_duration,
                "padding": self.segmentation.padding,
                "energy_threshold": self.segmentation.energy_threshold,
            },
            "grouping": {
                "method": self.grouping.method,
                "threshold": self.grouping.threshold,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
