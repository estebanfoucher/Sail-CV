"""
Video utilities module - provides convenience functions and imports for backward compatibility.

This module re-exports classes from the main video module to maintain compatibility
with existing code while the main video functionality has been moved to video.py.
"""

# Re-export main video classes for backward compatibility
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from video import Video, VideoReader, StereoVideoReader, FFmpegVideoWriter, FrameExtractor, get_unique_video_name

# Keep the module clean - all functionality is now in video.py
__all__ = [
    "Video",
    "VideoReader", 
    "StereoVideoReader",
    "FFmpegVideoWriter",
    "FrameExtractor",
    "get_unique_video_name"
]