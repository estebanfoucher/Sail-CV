"""
Video utilities module - provides convenience functions and imports for backward compatibility.

This module re-exports classes from the main video module to maintain compatibility
with existing code while the main video functionality has been moved to video.py.
"""

# Re-export main video classes for backward compatibility
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from video import (
    FFmpegVideoWriter,
    FrameExtractor,
    StereoVideoReader,
    Video,
    VideoReader,
    get_unique_video_name,
)

# Keep the module clean - all functionality is now in video.py
__all__ = [
    "FFmpegVideoWriter",
    "FrameExtractor",
    "StereoVideoReader",
    "Video",
    "VideoReader",
    "get_unique_video_name",
]
