"""Video processing modules."""

from drumcut.video.clip import ClipInfo, process_clip, process_session_clips
from drumcut.video.filters import FILTER_PRESETS, apply_filter
from drumcut.video.gopro import group_by_session, parse_gopro_filename
from drumcut.video.merge import merge_chapters
from drumcut.video.sync import sync_audio_to_video

__all__ = [
    "ClipInfo",
    "FILTER_PRESETS",
    "apply_filter",
    "group_by_session",
    "merge_chapters",
    "parse_gopro_filename",
    "process_clip",
    "process_session_clips",
    "sync_audio_to_video",
]
